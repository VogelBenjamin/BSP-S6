#include <stdio.h>
#include <cuda.h>
__global__ void c_function()
{
    int id = threadIdx.x;
    printf("Hello world!: %d \n",id);
    __syncthreads();
}

int main()
{
    c_function<<<1,10>>>();
    cudaDeviceSynchronize();
    int count = 0;
    for (int i = 0; i < 1e6; ++i)
    {
        count += 1;
    }
    printf("%d\n",count);
    return 0;
}