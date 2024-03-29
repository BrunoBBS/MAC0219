#ifndef __ACC_CUH__
#define __ACC_CUH__

#include <curand_kernel.h>

__global__ void gpu_calc(curandState_t *states, double *sum, double *sum_2,
                         uint64_t n_ops_thread, int64_t M, int64_t k,
                         uint64_t n_threads, uint64_t n_leap_ops);

__device__ double gpu_f(int64_t M, int64_t k, double x);


__device__ double atomicAdd(double* address, double val, double dummy);
#endif