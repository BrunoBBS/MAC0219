#include <cooperative_groups.h>

#include "kernel.cuh"

using namespace cooperative_groups;

__global__
void reduce_min(int n_mat, int* g_values)
{
    // Local(block) thread id
    unsigned int lid = threadIdx.x;

    // Global (across blocks) thread id
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Reference to this thread block
    thread_block this_block = this_thread_block();
    int *local_data = SharedMemory();

    local_data[lid] = (i < n_mat) ? g_values[i] : g_values[0];

    // wait untill all threads updated the array values
    sync(this_block);

    //reduce the vector
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (lid < s)
            local_data[lid] += min(local_data[lid], local_data[lid + s]);

        sync(this_block);
    }
    // Write results to global memory
    if (lid == 0) g_values[blockIdx.x * blockDim.x] = local_data[0];
}

void reduce(int num_mat, void* device_array)
{
    const int32_t block_size = 1024;
    const int32_t num_blocks = (num_mat / (block_size * 2)) + (num_mat % (block_size * 2) != 0);

    for (int item = 0; item < 9; item++)
        reduce_min<<<num_blocks, block_size>>>(num_mat, ((int*) device_array) + item * num_mat);
}