__global__
void reduce_min(int mat_num, int )
{
    int i = blockIdx*blockDim + threadIdx.x;
    if (i % 2 == 0) 
}