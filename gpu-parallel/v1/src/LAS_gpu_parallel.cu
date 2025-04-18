#include"LAS_gpu_parallel.h"
#include<cuda.h>

#include<stdio.h>
#include<stdlib.h>

// assume 1d Grid and 1d Block
#define TILE_SIZE 256

__global__ void matrix_vector_mult(unsigned int size, float* matrix, float* vector, float* result)
{
    __shared__ float vector_tile[TILE_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;

    if (row < size) {
        // Each thread processes the entire vector in tiles
        for (int tile_base = 0; tile_base < size; tile_base += TILE_SIZE) {
            int tile_end = min(tile_base + TILE_SIZE, size);
            
            // Cooperative loading of vector tile (all threads work together)
            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                vector_tile[i] = (tile_base + i < size) ? vector[tile_base + i] : 0.0f;
            }
            __syncthreads();

            // Compute partial sum for this tile
            for (int j = 0; j < TILE_SIZE; j++) {
                int col = tile_base + j;
                if (col < size) {
                    acc += matrix[row * size + col] * vector_tile[j];
                }
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
