#ifndef KERNEL_CU
#define KERNEL_CU
#include <cuda_runtime.h>

#define R_BLOCK_SZ 1024
#define C_BLOCK_SZ 1024

__global__ void reduce_min(unsigned int n_mat, int *g_values);

void reduce(unsigned int num_mat, void *device_array, unsigned int itemcnt);

#endif