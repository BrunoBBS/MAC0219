#include <cooperative_groups.h>

#include <stdio.h>

#include "kernel.cuh"

using namespace cooperative_groups;

__global__ void reduce_min(unsigned int n_mat, int* g_values)
{
    // Local(block) thread id
    unsigned int lid = threadIdx.x;

    // Global (across blocks) thread id
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Reference to this thread block
    thread_block this_block = this_thread_block();
    __shared__ int local_data[R_BLOCK_SZ * 2];

    local_data[lid] = (i < n_mat) ? g_values[i] : g_values[0];
    local_data[lid + blockDim.x] = (i + blockDim.x < n_mat) ? g_values[i + blockDim.x] : g_values[0];

    // wait until all threads updated the array values
    sync(this_block);

    //reduce the vector
    for (unsigned int s = blockDim.x; s > 0; s >>= 1)
    {
        if (lid < s)
            local_data[lid] = min(local_data[lid], local_data[lid + s]);

        sync(this_block);
    }

    // Write results to global memory
    if (lid == 0)
        g_values[2 * blockIdx.x * blockDim.x] = local_data[0];
}

__global__ void compress(int n_values, int* g_values, int interval)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n_values)
        g_values[gid] = g_values[interval * gid];
}

void reduce(unsigned int num_mat, void* device_array, unsigned int itemcnt)
{
    // Set the block size
    const int32_t block_size_r = R_BLOCK_SZ;
    const int32_t block_size_c = C_BLOCK_SZ;

    // We start with num_mat elements
    int elements = num_mat;

    int32_t num_blocks_r, num_blocks_c;

    do
    {
        // Calculates number of blocks sized block_size based on number of elements
        num_blocks_r = (elements / (block_size_r * 2)) + (elements % (block_size_r * 2) != 0);

        // Calls the kernel for each position of the matrix (piece of the vector)
        for (int item = 0; item < itemcnt; item++)
            reduce_min<<<num_blocks_r, block_size_r>>>(elements, ((int*) device_array) + item * num_mat);

        num_blocks_c = (num_blocks_r / block_size_c) + (num_blocks_r % block_size_c != 0);

        // Calls compressor kernel for global array
        for (int item = 0; item < itemcnt; item++)
            compress<<<num_blocks_c, block_size_c>>>(num_blocks_r, ((int*) device_array) + item * num_mat, block_size_r * 2);

        // Repeat with this iteration's results
        elements = num_blocks_r;
    } while (num_blocks_r > 1);

    // Compress one last time so that items are contiguous in device vector
    num_blocks_c = (itemcnt / block_size_c) + (itemcnt % block_size_c != 0);
    compress<<<num_blocks_c, block_size_c>>>(itemcnt, (int*) device_array, num_mat);
}