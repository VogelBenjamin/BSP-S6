#include"LAS_gpu_parallel.h"
#include<cuda.h>

#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

// assume 1d Grid and 1d Block
__global__ void matrix_vector_mult(unsigned int size, float* matrix, float* vector, float* vector_storage)
{
	int row = blockDim.x*blockIdx.x + threadIdx.x;
  //int col = blockDim.y*blockIdx.y + threadIdx.y

  if (row < size)
  {
      float acc = 0;
      for (int i = 0; i < size; ++i)
      {
          acc += matrix[row*size+i]*vector[row];
      }
      vector_storage[row] = acc;
  }
	__syncthreads();
	return;
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

void compute_residual_gpu(unsigned int size, float* A, float* b, float* x, float* vector_storage, dim3 GridDim, dim3 BlockDim)
{
	matrix_vector_mult<<<GridDim,BlockDim>>>(size,A,x,vector_storage);
	vector_sub(size,vector_storage,b,1.0,vector_storage);
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
