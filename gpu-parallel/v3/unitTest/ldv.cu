#include <stdio.h>
#include <cuda.h>
#include"../../matrices/matrix_load.h"

__global__ void vector_add(float *a, float *b, float *out, int N)
{
    int id = blockIdx.x * blockDim.x * blockDim.y +
             threadIdx.y * blockDim.x +
             threadIdx.x;
    if(id < N){
        out[id] += a[id] + b[id];
    }

    __syncthreads();
}

int main()
{
    int N = 420;
    float *a, *b, *out;

    a = load_FFGE_float("../../matrices/matrix_data/bcsstk06.mtx");
    b = load_FFGE_float("../../matrices/matrix_data/bcsstk06.mtx");
    out = (float*)malloc(sizeof(float)*N*N);

    float *d_a, *d_b, *d_out;

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N*N);
    cudaMalloc((void**)&d_b, sizeof(float) * N*N);
    cudaMalloc((void**)&d_out, sizeof(float) * N*N);


    cudaMemcpy(d_a, a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    // Thread organization
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);

    vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, N*N);

    cudaMemcpy(out, d_out, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N*N; ++i)
    {
        printf("out: %f\n", out[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
    return 0;
}
