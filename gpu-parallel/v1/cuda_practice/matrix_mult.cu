#include<stdio.h>
#include<cuda.h>

__global__ void matrix_mult(float* a, float* b, float* c, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;


    if (row < N && col < N) {
         float temp = 0;
         for (int i = 0; i < N; ++i)
        {
            temp += a[row*N+i]*b[i*N+col];
        }
         c[row*N+col] = temp;
    }
    __syncthreads();
}

int main()
{
    int N = 10;
    float *a, *b, *c;

    a = (float*)malloc(sizeof(float)*N*N);
    b = (float*)malloc(sizeof(float)*N*N);
    c = (float*)malloc(sizeof(float)*N*N);

    float *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, sizeof(float) * N*N);
    cudaMalloc((void**)&d_b, sizeof(float) * N*N);
    cudaMalloc((void**)&d_c, sizeof(float) * N*N);

    for(int i = 0; i < N*N; i++)
    {
      a[i] = 1.0;
      b[i] = 2.0;
    }

    cudaMemcpy(d_a, a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
    matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%f ", c[i*N+j]);
        }
        printf("\n");
    }
    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Deallocate host memory
    free(a);
    free(b);
    free(c);
    return 0;
}
