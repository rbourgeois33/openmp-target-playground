// ============================================================
//  2D Heat Equation  –  Explicit Finite Difference
//  Multi-kernel benchmark harness
// ============================================================

//  nvc++ -O3 -march=native -Minfo -mp=gpu -gpu=cc89 -fast  -cuda -lnvToolsExt  -o main main.cpp 
// -cuda for cuda kernel
// OMP_PROC_BIND=true OMP_NUM_THREADS=24  ./main
// OMP_PROC_BIND=true OMP_NUM_THREADS=24  ./main --no-sequential
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <nvToolsExt.h>

// ---------- parameters --------------------------------------
static constexpr int    NX          = 4096;
static constexpr int    NY          = 2048;
static constexpr double LX          = 1.0;
static constexpr double LY          = 1.0;
static constexpr double ALPHA       = 0.01;
static constexpr int    NSTEP       = 2000;
static constexpr int    PRINT_EVERY = 250;
const long long N      = (long long)NX * NY;


// ---------- derived -----------------------------------------
static constexpr double DX  = LX / (NX - 1);
static constexpr double DY  = LY / (NY - 1);
static constexpr double DT  = 0.24 * DX * DX / ALPHA;
static constexpr double RX  = ALPHA * DT / (DX * DX);
static constexpr double RY  = ALPHA * DT / (DY * DY);

inline int idx(int i, int j) { return i * NY + j; }

inline double analytical(double x, double y, double t)
{
    return std::exp(-2.0 * M_PI * M_PI * ALPHA * t)
         * std::sin(M_PI * x) * std::sin(M_PI * y);
}

void initialise(double* u, double t)
{
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            u[idx(i,j)] = analytical(i*DX, j*DY, t);
}

double l2_error(const double* u, double t)
{
    double err2 = 0.0;
    for (int i = 1; i < NX-1; ++i)
        for (int j = 1; j < NY-1; ++j) {
            double d = u[idx(i,j)] - analytical(i*DX, j*DY, t);
            err2 += d * d;
        }
    return std::sqrt(err2 / ((NX-2)*(NY-2)));
}

// ============================================================
//  K E R N E L S
// ============================================================

//base seq
void sequential(const double* __restrict__ u,
                      double* __restrict__ un,
                int nx, int ny, double rx, double ry, bool on_gpu)
{
    for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j) { //Generated vector simd code for the loop
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// Parallelizes only the outer i loop across CPU threads. Good cache reuse.
// Best CPU perfs (x19)
// When a thread encounters a parallel construct, a team of threads is created to execute the parallel
// An implicit barrier occurs at the end of a parallel region. After the end of a parallel region, only the primary thread of the team resumes execution of the enclosing task region.
void openmp_parallel_for(const double* __restrict__ u,
                               double* __restrict__ un,
                         int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp parallel for
    for (int i = 1; i < nx-1; ++i) //#omp parallel
        for (int j = 1; j < ny-1; ++j) { //Generated vector simd code for the loop
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// Collapses both loops into one large iteration space.
// Less cache reuse since we force the compiler to parallelize more.
//CPU perf= x10
void openmp_parallel_for_collapse(const double* __restrict__ u,
                                        double* __restrict__ un,
                                  int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp parallel for collapse(2) //#omp parallel
    for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}


// GPU: Poor performance(x5). 142blocks of 128 threads ? not sure whats going on. Poor maping of parallel for on GPU
// It seems from "Loop parallelized across threads(128), schedule(static)" that we launch 1 block
//      Small D2H/H2D between kernel ? Yes
// CPU: Poor perf (x1). No idea why not the same tha, openmp_parallel_for. kills // target and parallel for
void openmp_target_parallel_for(const double* __restrict__ u,
                                      double* __restrict__ un,
                                int nx, int ny, double rx, double ry, bool on_gpu)
{ 
    #pragma omp target parallel for if (on_gpu)
    for (int i = 1; i < nx-1; ++i) //GPU: Loop parallelized across threads(128), schedule(static). Loop not vectorized/parallelized: not countable. CPU: Loop parallelized across threads, schedule(static)
        for (int j = 1; j < ny-1; ++j) {  //GPU: Generated vector simd code for the loop
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Poor performance(x5). 142blocks of 128 threads ? not sure whats going on. Poor maping of parallel for on GPU
//      Small D2H/H2D between kernel ? Yes
// CPU: Poor perf (x1). No idea why not the same tha, openmp_parallel_for. kills // target and parallel for
void openmp_target_parallel_for_collapse(const double* __restrict__ u,
                                               double* __restrict__ un,
                                         int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target parallel for collapse(2) if (on_gpu)
    for (int i = 1; i < nx-1; ++i) // GPU Loop parallelized across threads(128), schedule(static) Loop not vectorized/parallelized: not countable Generated vector simd code for the loop. CPU: Loop parallelized across threads, schedule(static)
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Poor performance(x8). 32 blocks of 128 threads = NX threads, only outer//
//      Small D2H/H2D between kernel ? NO
// CPU: Okay perfs perf(x11). Looks like  openmp_parallel_for_collapse 
// When a thread encounters a teams construct, a league of teams is created. if: one team
//A loop construct speciﬁes that the logical iterations of the associated loops may execute concurrently and permits the encountering threads to execute the loop accordingly
//Team sert un peu a rien loop peut spawn des team
void openmp_target_team_loop(const double* __restrict__ u,
                                   double* __restrict__ un,
                             int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams loop if (on_gpu)
    for (int i = 1; i < nx-1; ++i) // GPU: Loop parallelized across teams, threads(128); Loop parallelized across threads
        for (int j = 1; j < ny-1; ++j) { //Loop run sequentially // Loop carried dependence of un-> prevents parallelization
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Full performance(x36). NX=4096 blocks of 128 thread. Fully //
//      Small D2H/H2D between kernel ? NO
// CPU: Great perf(x18)
void openmp_target_team_loop_collapse(const double* __restrict__ u,
                                            double* __restrict__ un,
                                      int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams loop collapse(2) if (on_gpu)
    for (int i = 1; i < nx-1; ++i) // Loop parallelized across teams, threads(128) collapse(2) Loop parallelized across threads
        for (int j = 1; j < ny-1; ++j) { //Generated vector simd code for the loop
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}


// GPU: Bad performance(x1.8). NX=4096 blocks of 128 thread but no inner //
//      Small D2H/H2D between kernel ? YES
// CPU: Same as seq(x1). teams+parallel_for=no // on CPU ?
// The distribute construct speciﬁes that the iterations of one or more loops will be executed by the initial teams in the context of their implicit tasks. The iterations are distributed across the initial threads of all initial teams that execute the teams region to which the distribute region binds. No implicit barrier occurs at the end of a distribute region
void openmp_target_team_distribute(const double* __restrict__ u,
                                         double* __restrict__ un,
                                   int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams distribute if (on_gpu)
    for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}


//Distributr alone without parallel is TERRIBLE, it lanches blocks that do no inner //
void openmp_target_team_distribute_collapse(const double* __restrict__ u,
                                         double* __restrict__ un,
                                   int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams distribute collapse(2) if (on_gpu)
    for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Not Full performance(x5). 32 blocks of 128 threads. 32*128=4096=NX. Only outer is //
//      Small D2H/H2D between kernel ? NO
// CPU: Same as seq(x1). if kills  target parallel
void openmp_target_team_distribute_for(const double* __restrict__ u,
                                             double* __restrict__ un,
                                       int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams distribute parallel for if (on_gpu)
    for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Full performance(x36). NX blocks of 128 threads
//      Small D2H/H2D between kernel ? YES
// Faster than distribute alone: threads within a block parallelize the inner loop.
// CPU: Very slow(x0.16). Not sure why. Teams not for CPU ? i seq, j //, we pay nxth time the trhead spawn overhead ?
void openmp_target_team_distribute_nestedfor(const double* __restrict__ u,
                                                   double* __restrict__ un,
                                             int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams distribute if (on_gpu)
    for (int i = 1; i < nx-1; ++i)
        #pragma omp parallel for
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Full performance(x36). Launches NX*NY threads; explicit full grid via collapse.
//      Small D2H/H2D between kernel ? NO
// CPU: Very slow(x0.3). Not sure why. Teams not for CPU ?
void openmp_target_team_distribute_for_collapse(const double* __restrict__ u,
                                                      double* __restrict__ un,
                                                int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target teams distribute parallel for collapse(2) if (on_gpu)
    for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j) {
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

// GPU: Poor performance(x8). 32 blocks of 128 threads = NX threads, only outer//
//      Small D2H/H2D between kernel ? NO
void openmp_target_loop(const double* __restrict__ u,
                                   double* __restrict__ un,
                             int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target loop if (on_gpu)
    for (int i = 1; i < nx-1; ++i) 
        for (int j = 1; j < ny-1; ++j) { 
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

//Full perf CPU et GPU !!
void openmp_target_loop_collapse(const double* __restrict__ u,
                                   double* __restrict__ un,
                             int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp target loop collapse(2) if (on_gpu)
    for (int i = 1; i < nx-1; ++i) 
        for (int j = 1; j < ny-1; ++j) { 
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}

//Also possible with metadirective to e.g. not desacritavte parallel on CPU with a if that only should deactivate target
//from: https://passlab.github.io/Examples/contents/Chap_program_control/7_Metadirectives.html
//also possible:
// when(     device={arch("nvptx")}:nvptx architecture is active in the OpenMP context
// when( implementation={vendor(nvidia)}, device={arch("kepler")}: when clause to distinguish between platforms.
// when( implementation={vendor(amd)},device={arch("fiji"  ): when clause to distinguish between platforms.
// when(   construct={target}:: are you in a target region ?
void openmp_metadirective(const double* __restrict__ u,
                                   double* __restrict__ un,
                             int nx, int ny, double rx, double ry, bool on_gpu)
{
    #pragma omp metadirective \
        when( user={condition(on_gpu)}: target teams distribute parallel for collapse(2)) \
        default(parallel for collapse(2))     
    for (int i = 1; i < nx-1; ++i) 
        for (int j = 1; j < ny-1; ++j) { 
            double center = u[idx(i,j)];
            double lap = (u[idx(i-1,j)] - 2.0*center + u[idx(i+1,j)]) * rx
                       + (u[idx(i,j-1)] - 2.0*center + u[idx(i,j+1)]) * ry;
            un[idx(i,j)] = center + lap;
        }
}


__global__
void cuda_kernel(const double* __restrict__ u,
                 double* __restrict__ un,
                 int nx, int ny,
                 double rx, double ry)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // skip boundaries
    if (i >= 1 && i < nx - 1 &&
        j >= 1 && j < ny - 1)
    {
        double center = u[idx(i, j)];

        double lap =
            (u[idx(i - 1, j)] - 2.0 * center + u[idx(i + 1, j)]) * rx +
            (u[idx(i, j - 1)] - 2.0 * center + u[idx(i, j + 1)]) * ry;

        un[idx(i, j)] = center + lap;
    }
}

//Best of all but not cpu
void cuda(const double* __restrict__ u,
                                   double* __restrict__ un,
                             int nx, int ny, double rx, double ry, bool on_gpu)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((nx  + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    #pragma omp target data use_device_ptr(u,un) //necessary otherwise u and un are host vectors
    {
    cuda_kernel<<<gridDim, blockDim>>>(u, un, nx, ny, rx, ry);
    }
}



// ============================================================
//  Benchmark harness
// ============================================================
using StepFn = std::function<void(const double*, double*, int, int, double, double, bool)>;

struct Kernel {
    std::string name;
    StepFn      fn;
    bool        on_gpu;
};

struct Result {
    std::string name;
    double      elapsed_s;
    double      bw_GBs;
    double      gflops;
    double      final_l2;
    bool        stable;
    bool        on_gpu;
};

Result run_kernel(const Kernel& k, double *pu, double* pu_new)
{
    const long long Nint   = (long long)(NX-2)*(NY-2);
    const double bytes_per = (double)N * 2.0 * sizeof(double);

    double t = 0.0;
    nvtxRangePushA("Initialize");
    initialise(pu, t);
    nvtxRangePop();

    using clk = std::chrono::high_resolution_clock;
    double elapsed;
    auto t0 = clk::now();
    auto t1 = clk::now();

    nvtxRangePushA("Send To Device");
    #pragma omp target update to(pu[0:N]) if (k.on_gpu)
    nvtxRangePop();
    nvtxRangePushA("Steps");

    {
        t0 = clk::now();
        for (int n = 0; n < NSTEP; ++n) {
            nvtxRangePushA("step");
            k.fn(pu, pu_new, NX, NY, RX, RY, k.on_gpu);
            nvtxRangePop();
            t += DT;
            std::swap(pu, pu_new);
        }
        t1 = clk::now();
    }
    nvtxRangePop();
    nvtxRangePushA("Get From Device");
    #pragma omp target update from(pu[0:N]) if (k.on_gpu)
    nvtxRangePop();

    elapsed = std::chrono::duration<double>(t1-t0).count();

    nvtxRangePushA("Error calculation");
    double l2   = l2_error(pu, t);
    nvtxRangePop();

    double bw   = bytes_per * NSTEP / elapsed / 1e9;
    double gf   = 14.0 * Nint * NSTEP / elapsed / 1e9;
    bool stable = std::isfinite(l2) && (l2 < 1.0);

    return { k.name, elapsed, bw, gf, l2, stable, k.on_gpu };
}

// ============================================================
int main(int argc, char** argv)
{

    printf("Grid: %dx%d   DT=%.4e   RX=%.4f   RY=%.4f\n",
           NX, NY, DT, RX, RY);

    // ---- register kernels (name, fn, on_gpu) -
    std::vector<Kernel> kernels = {
        { "sequential",                              sequential,                              false},
        { "openmp_parallel_for",                     openmp_parallel_for,                     false},
        { "openmp_parallel_for_collapse",            openmp_parallel_for_collapse,            false},
        { "openmp_target_parallel_for",              openmp_target_parallel_for,              true},
        { "openmp_target_parallel_for",              openmp_target_parallel_for,              false},
        { "openmp_target_parallel_for_collapse",     openmp_target_parallel_for_collapse,     true},
        { "openmp_target_parallel_for_collapse",     openmp_target_parallel_for_collapse,     false},
        { "openmp_target_team_loop",                 openmp_target_team_loop,                 true},
        { "openmp_target_team_loop",                 openmp_target_team_loop,                 false},
        { "openmp_target_team_loop_collapse",        openmp_target_team_loop_collapse,        true},
        { "openmp_target_team_loop_collapse",        openmp_target_team_loop_collapse,        false},
        { "openmp_target_team_distribute",           openmp_target_team_distribute,           true},
        { "openmp_target_team_distribute",           openmp_target_team_distribute,           false},
        { "openmp_target_team_distribute_collapse",           openmp_target_team_distribute_collapse,           true},
        { "openmp_target_team_distribute_collapse",           openmp_target_team_distribute_collapse,           false},
        { "openmp_target_team_distribute_for",       openmp_target_team_distribute_for,       true},
        { "openmp_target_team_distribute_for",       openmp_target_team_distribute_for,       false},
        { "openmp_target_team_distribute_nestedfor", openmp_target_team_distribute_nestedfor, true},
        { "openmp_target_team_distribute_nestedfor", openmp_target_team_distribute_nestedfor, false},
        { "openmp_target_team_distribute_for_collapse", openmp_target_team_distribute_for_collapse, true},
        { "openmp_target_team_distribute_for_collapse", openmp_target_team_distribute_for_collapse, false},
        { "openmp_target_loop", openmp_target_loop, true},
        { "openmp_target_loop", openmp_target_loop, false},
        { "openmp_target_loop_collapse", openmp_target_loop_collapse, true},
        { "openmp_target_loop_collapse", openmp_target_loop_collapse, false},
        { "openmp_metadirective", openmp_metadirective, true},
        { "openmp_metadirective", openmp_metadirective, false},

        { "cuda", cuda, true},

    };

    // ---- compute column width from longest kernel name ------
    int name_w = 8; // minimum: len("kernel  ")
    for (const Kernel& k : kernels)
        name_w = std::max(name_w, (int)k.name.size());
    // device tag "[GPU]" / "_cpu_" / "[skipped]" get their own fixed column
    const int tag_w   = 9; // len("[skipped]")
    const int total_w = name_w + 2 + tag_w + 2 + 8 + 2 + 10 + 2 + 10 + 2 + 12 + 2 + 4;

    // ---- benchmark all (non-skipped) kernels ----------------
    printf("\n%s\n", std::string(total_w, '=').c_str());
    printf("%-*s  %-*s  %8s  %10s  %10s  %12s  %s\n",
           name_w, "kernel", tag_w, "device",
           "time(s)", "BW(GB/s)", "GFLOP/s", "L2_err", "ok?");
    printf("%s\n", std::string(total_w, '-').c_str());

    Result best_bw{};
    best_bw.bw_GBs = 0.0;

    std::vector<Result> results;
    results.reserve(kernels.size());

    nvtxRangePushA("Alloc CPU");
    std::vector<double> u    (N, 0.0);
    std::vector<double> u_new(N, 0.0);
    nvtxRangePop();

    double* pu     = u.data();
    double* pu_new = u_new.data();

    // Keep the sequential result for speedup reference even if skipped
    double t_ref = -1.0;

    nvtxRangePushA("Alloc GPU");
    //The target data construct maps variables to a device data environment. When a target data construct is encountered, the encountering task executes the region. When an if clause is present and the if clause expression evaluates to false, the target device is the host.
    #pragma omp target enter data map(alloc: pu[0:N], pu_new[0:N]) //enter / exit are scope less ! no nee to {}. pu and u are on the gpu until exit
    nvtxRangePop(); //For scoping, use omp target data

    for (const Kernel& k : kernels) {
        nvtxRangePushA(k.name.c_str());
        Result r = run_kernel(k, pu, pu_new);
        nvtxRangePop();

        if (k.name=="sequential")
            t_ref = r.elapsed_s;

        results.push_back(r);
        printf("%-*s  %-*s  %8.3f  %10.2f  %10.3f  %12.4e  %s\n",
            name_w, r.name.c_str(),
            tag_w,  r.on_gpu ? "[GPU]" : "_cpu_",
            r.elapsed_s, r.bw_GBs, r.gflops,
            r.final_l2, r.stable ? "PASS" : "FAIL");

        if (r.bw_GBs > best_bw.bw_GBs)
            best_bw = r;
    }

    nvtxRangePushA("Free GPU");
    #pragma omp target exit data map(delete: pu[0:N], pu_new[0:N])
    nvtxRangePop();


    printf("%s\n", std::string(total_w, '=').c_str());

    // ---- relative speedup vs sequential ---------------------
    if (t_ref > 0.0) {
        printf("\nSpeedup vs 'sequential':\n");
        for (const Result& r : results)
            printf("  %-*s  %-*s  %.2fx\n",
                   name_w, r.name.c_str(),
                   tag_w,  r.on_gpu ? "[GPU]" : "_cpu_",
                   t_ref / r.elapsed_s);
    } else {
        printf("\n(speedup vs sequential not available: sequential was skipped)\n");
    }

    printf("\nBest BW: %-*s  %-*s  %.2f GB/s\n",
           name_w, best_bw.name.c_str(),
           tag_w,  best_bw.on_gpu ? "[GPU]" : "_cpu_",
           best_bw.bw_GBs);

    return 0;
}