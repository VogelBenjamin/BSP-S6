#include"LAS_gpu_parallel.h"
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
			acc += matrix[row*size+i]*vector[i];
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

__global__ void dot_product(unsigned int size, float* vector_1, float* vector_2, float* res)
{
	__shared__ float block_sum[256];  // Max 256 threads per block
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // Initialize local product
    float product = 0.0f;
    if (tid < size) {
        product = vector_1[tid] * vector_2[tid];
    }

    // Store in shared memory
    block_sum[local_idx] = product;
    __syncthreads();

    // Reduce within block (simple sequential reduction for readability)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            block_sum[local_idx] += block_sum[local_idx + stride];
        }
        __syncthreads();
    }

    // Add block sum to global total
    if (local_idx == 0) {
        atomicAdd(res, block_sum[0]);
    }
}

__global__ void vector_add(unsigned int size, float* vector_1, float* vector_2, float* num, float* denum, float* vector_storage)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size)
	{
		vector_storage[idx] = vector_1[idx]+((*num)/(*denum))*vector_2[idx];
	}
	return;
}

__global__ void vector_sub(unsigned int size, float* vector_1, float* vector_2, float* num, float* denum, float* vector_storage)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size)
	{
		vector_storage[idx] = vector_1[idx]-((*num)/(*denum))*vector_2[idx];
	}
}

__global__ void vector_copy(unsigned int size, float* vector_out, float* vector_in)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size)
	{
		vector_out[idx] = vector_in[idx];
	}
	return;
}

__global__ void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < size)
    {
	    vector[idx] *= alpha;
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
