#ifndef KERNEL_CU
#define KERNEL_CU
#include <cuda_runtime.h>

struct SharedMemory
{
    __device__ inline operator int *()
    {
        extern __shared__ int __smem[];
        return __smem;
    }

    __device__ inline operator const int *() const
    {
        extern __shared__ int __smem[];
        return __smem;
    }
};

__global__ void reduce_min(unsigned int n_mat, int *g_values);

void reduce(unsigned int num_mat, void *device_array, unsigned int itemcnt);

#endif