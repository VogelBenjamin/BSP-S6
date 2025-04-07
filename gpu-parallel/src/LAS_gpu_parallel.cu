#include<LAS_gpu_parallel.h>
#include<cuda.h>

#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

// assume 1d Grid and 1d Block
__global__ void matrix_vector_mult(unsigned int size, double* matrix, double* vector, double* vector_storage)
{
	int row = blockDim.x*blockIdx.x + threadIdx.x
  //int col = blockDim.y*blockIdx.y + threadIdx.y

  if (row < size && col < size)
  {
      float acc = 0
      for (int i = 0; i < size; ++i)
      {
          acc += matrix[row*size+i]*vector[row];
      }
      vector_storage = acc;
  }
  __synchthreads();
	return;
}

double dot_product(unsigned int size, double* vector_1, double* vector_2)
{
	double storage = 0;
	for (unsigned int i = 0; i < size; ++i)
	{
		storage += vector_1[i]*vector_2[i]; 
	}
	return storage;
}

void vector_add(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		vector_storage[i] = vector_1[i] + alpha*vector_2[i];	
	}
	return;
}

void vector_sub(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage)
{
        for (unsigned int i = 0; i < size; ++i)
        {
                vector_storage[i] = vector_1[i] - alpha*vector_2[i];
        }
        return;
}

void vector_copy(unsigned int size, double* vector_out, double* vector_in)
{
	for (unsigned int i = 0; i < size; ++i)
        {
                vector_out[i] = vector_in[i];
        }
	return;
}

void compute_residual(unsigned int size, double* A, double* b, double* x, double* vector_storage)
{
	matrix_vector_mult(size,A,x,vector_storage);
	vector_sub(size,vector_storage,b,1.0,vector_storage);
	return;
}

void scalar_vector_mult_inplace(unsigned int size,double* vector, double alpha)
{
	for (unsigned int i = 0; i < size; ++i)
	{
			vector[i] = alpha*vector[i];
	}
    return;
}

void print_vector(unsigned int size, double* vector)
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