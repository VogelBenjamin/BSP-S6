#include"LAS_gpu_parallel.h"
#include<cuda.h>

#include<stdio.h>
#include<stdlib.h>

#define TILE_SIZE 256

__global__ void matrix_vector_mult(unsigned int size, float* matrix, float* vector, float* result)
{
    __shared__ float vector_tile[TILE_SIZE];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;

    if (row < size) {
        // Each thread computes one element of the result vector
        for (int tile_start = 0; tile_start < size; tile_start += TILE_SIZE) {
            int tile_end = min(tile_start + TILE_SIZE, size);
            int valid_tile_size = tile_end - tile_start;

            // Cooperative loading of vector tile into shared memory
            for (int i = threadIdx.x; i < valid_tile_size; i += blockDim.x) {
                vector_tile[i] = vector[tile_start + i];
            }
            __syncthreads();

            // Compute partial sum for this tile
            for (int j = 0; j < valid_tile_size; j++) {
                acc += matrix[row * size + (tile_start + j)] * vector_tile[j];
            }
            __syncthreads();
        }
        result[row] = acc;
    }
}


__global__ void test_access(float* vec) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx == 0) printf("vec[0] = %f\n", vec[0]);
}

float dot_product(unsigned int size, float* vector_1, float* vector_2)
{
	float storage = 0;
	for (unsigned int i = 0; i < size; ++i)
	{
		storage += vector_1[i]*vector_2[i]; 
	}
	return storage;
}

void vector_add(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		vector_storage[i] = vector_1[i] + alpha*vector_2[i];	
	}
	return;
}

void vector_sub(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage)
{
        for (unsigned int i = 0; i < size; ++i)
        {
                vector_storage[i] = vector_1[i] - alpha*vector_2[i];
        }
        return;
}

void vector_copy(unsigned int size, float* vector_out, float* vector_in)
{
	for (unsigned int i = 0; i < size; ++i)
        {
                vector_out[i] = vector_in[i];
        }
	return;
}

void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha)
{
	for (unsigned int i = 0; i < size; ++i)
	{
			vector[i] = alpha*vector[i];
	}
    return;
}

void print_vector(unsigned int size, float* vector)
{
	for (unsigned int i = 0; i < size; ++i)
	{
			printf("%lf ",vector[i]);
	}
	printf("\n");
}


/*
// assume 1d Grid and 1d Block
__global__ void dot_product(unsigned int size, float* vector_1, float* vector_2, float* vector_out)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < size)
  {
	  vector_out[i] = vector_1[i]*vector_2[i];
  }
  __syncthreads();
}

__global__ void vector_add(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < size)
  {
	  vector_out[i] = vector_1[i]+vector_2[i];
  }
  __syncthreads();
}

__global__ void vector_sub(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < size)
  {
	  vector_out[i] = vector_1[i]-vector_2[i];
  }
  __syncthreads();
}

__global__ void vector_copy(unsigned int size, float* vector_out, float* vector_in)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < size)
  {
	  vector_out[i] = vector_in[i];
  }
  __syncthreads();
}

__global__ void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < size)
  {
	  vector[i] = alpha*vector[i];
  }
  __syncthreads();
}
*/
