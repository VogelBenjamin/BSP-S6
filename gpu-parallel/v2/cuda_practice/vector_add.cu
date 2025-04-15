#include <stdio.h>
#include <cuda.h>
#include "vector_add.h"

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

void exec()
{
    int N = 10;
    float *a, *b, *out;

    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float)*N);
    out = (float*)malloc(sizeof(float)*N);

    float *d_a, *d_b, *d_out;

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    for(int i = 0; i < N; i++)
    {
      a[i] = 1.0;
      b[i] = 2.0;
    }


    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    // Thread organization
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(16, 16, 1);

    vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, N);

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        printf("out: %f\n", out[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}
/*
int main()
{
    int N = 10;
    float *a, *b, *out;

    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float)*N);
    out = (float*)malloc(sizeof(float)*N);

    float *d_a, *d_b, *d_out;

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    for(int i = 0; i < N; i++)
    {
      a[i] = 1.0;
      b[i] = 2.0;
    }


    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    // Thread organization
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(16, 16, 1);

    vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, N);

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
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
    */
