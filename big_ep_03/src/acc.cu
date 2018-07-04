#include "acc.cuh"
#include "gpu.hpp"
#include <cmath>
#include <climits>
#include "util.hpp"
#include <cuda_runtime.h>
#include <chrono>


using namespace std::chrono;

/* This code was copied from
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 * since there is no atomicAdd for GPUs with Compute Capability less than 6.0.
 * The dummy parameter fixes any overload or double definition errors.
 */
__device__ double atomicAdd(double* address, double val, double dummy)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}

__host__ double gpu_probing(uint64_t n_ops, int64_t M, int64_t k)
{

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    gpu_integration(n_ops, M, k);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> dur = duration_cast<duration<double>>(t2 - t1);
    double flops         = n_ops / dur.count();
    return flops;
}

__device__ double gpu_f(int64_t M, int64_t k, double x)
{
    return (sin((2 * M + 1) * M_PI * x) * cos(2 * M_PI * k * x)) /
           sin(M_PI * x);
}

/** Calculates f for n_ops points and puts the sum in the global memory
 */
__global__ void gpu_calc(curandState_t *states, double *sum, double *sum_sq,
                         uint64_t n_ops_thread, int64_t M, int64_t k,
                         uint64_t n_threads, uint64_t n_leap_ops)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    __shared__ double block_sum[1024];
    __shared__ double block_sum_sq[1024];

    block_sum[lid] = block_sum_sq[lid] = 0;

    if (tid >= n_threads) return;

    double x, res;
    curandState_t local_state = states[tid];
    curand_init (tid, 0, 0, &local_state);

    n_ops_thread += tid < n_leap_ops;

    for (uint64_t i = 0; i < n_ops_thread; i++)
    {
        x   = curand_uniform_double(&local_state) / 2;
        res = gpu_f(M, k, x);
        block_sum[lid] += res;
        block_sum_sq[lid] += res * res;
    }

    __syncthreads();

    // Reduction phase
    for (int s = blockDim.x / 2; s > 0 && lid < s; s >>= 1)
    {
        block_sum[lid]   += block_sum[lid + s];
        block_sum_sq[lid] += block_sum_sq[lid + s];
        __syncthreads();
    }

    if (lid == 0)
    {
        atomicAdd(sum, block_sum[0], 0);
        atomicAdd(sum_sq, block_sum_sq[0], 0);
    }
}

__host__ std::vector<double> gpu_integration(uint64_t n_ops, int64_t M,
                                             int64_t k)
{

    if (!n_ops)
    {
        std::vector<double> res = {0, 0};
        return res;
    }

    int block_size = 1024;

    // Divide the work
    // The GPU has n_ops operations to do
    uint64_t n_threads    = min((uint64_t) ULONG_MAX * block_size, n_ops);
    uint64_t n_blocks     = ceil(n_threads / (float) block_size);
    uint64_t n_ops_thread = n_ops / n_threads;
    uint64_t n_leap_ops   = n_ops % n_threads;

    /*********************************
     * CUDA environment and device setup
     *********************************/
    // States for the random generator
    curandState_t *states;
    // Allocate space for the states
    cudaMalloc((void **)&states, n_ops * sizeof(curandState));
    // Pointer for the device global sum and sum_sq variables
    double *sum;
    // Allocate both variables as an array to save code lines :)
    cudaMalloc((void **)&sum, 2 * sizeof(double));
    // I want to pass two variables to gpu_calc and not an array
    double *sum_sq = sum + 1;
    // Zero the variables as they will be part of a summation
    cudaMemset(sum, 0, 2 * sizeof(double));

    // Actually call the calc function
    gpu_calc<<<n_blocks, block_size>>>(states, sum, sum_sq, n_ops_thread, M, k, n_threads, n_leap_ops);

    std::vector<double> sums(2);
    cudaMemcpy(sums.data(), sum, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(states);

    return sums;
}
