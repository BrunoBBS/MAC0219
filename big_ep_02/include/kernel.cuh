#ifndef KERNEL_CU
#define KERNEL_CU

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

__global__
void reduce_min(int n_mat, int* g_values);

#endif