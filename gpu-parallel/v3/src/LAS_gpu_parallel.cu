#include"LAS_gpu_parallel.h"
#include<cuda.h>

#include<stdio.h>
#include<stdlib.h>

#define TILE_SIZE 256
// assume 1d Grid and 1d Block
__global__ void matrix_vector_mult(unsigned int size, float* matrix, float* vector, float* vector_storage)
{
	__shared__ float cache[TILE_SIZE]; 

	int row = blockDim.x*blockIdx.x + threadIdx.x;
	int tidx = threadIdx.x;
	int bdim = blockDim.x;
	int elem_per_thread = (TILE_SIZE + blockDim.x - 1) / blockDim.x;
	
    if (row < size)
	{	
		
		float acc = 0.0f;

		for (int i = 0; i < (size+TILE_SIZE); i+= TILE_SIZE)
		{
			for (int j = 0; j < elem_per_thread; j++)
			{
				int idx = tidx+j*bdim;
				if (i+idx < size && idx < TILE_SIZE)
					cache[idx] = vector[i+idx];
			}
			__syncthreads();

			for (int j = 0; j < TILE_SIZE; ++j)
			{
				acc += matrix[row*size+i+j]*cache[j];
			}

			__syncthreads();
		}
		vector_storage[row] = acc;
		//printf("vector_storage[%d] = %f\n", row, acc);
	}
	
	return;
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
