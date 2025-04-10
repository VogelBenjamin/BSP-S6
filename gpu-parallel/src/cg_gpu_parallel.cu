#include"cg_gpu_parallel.h"
#include"LAS_gpu_parallel.h"
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>

#define CACHE_BLOCK_SIZE 64 // bytes
#define BLOCK_DIM 32
// compute step size


// check where to use 
extern "C" float* cg(unsigned int size, float* A, float* b, float* init_g, float epsilon, int debug)
{
	float*  solution;
	float*  residual;
	float*  residual_prev;
	float*  search_direction;
	float*  intermediate_comp;
	float alpha;
	float beta;
	float num;
	float denum;
	float err;
	int i = 0;
    printf("Size : %d \n", size);
	printf("Grid: %d , Block: %d\n", (size + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM);	
	dim3 GridDim( (size + BLOCK_DIM - 1) / BLOCK_DIM ,1,1);
	dim3 BlockDim(BLOCK_DIM,1,1);


	solution = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	residual = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	residual_prev = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	search_direction = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	intermediate_comp = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);



	scalar_vector_mult_inplace(size,solution,0);
	scalar_vector_mult_inplace(size,residual,0);
	scalar_vector_mult_inplace(size,residual_prev,0);
	scalar_vector_mult_inplace(size,search_direction,0);
	scalar_vector_mult_inplace(size,intermediate_comp,0);

	vector_copy(size, solution, init_g);
	
	compute_residual_gpu(size, A, b, solution, residual,GridDim,BlockDim);
	cudaDeviceSynchronize();
	vector_copy(size,search_direction,residual);
	
	scalar_vector_mult_inplace(size,search_direction,-1);
	
	err = dot_product(size, residual, residual);

	err = sqrt(err);

    	float* d_A, *d_sd, *d_ic;
    	//cudaMalloc((void**)&d_A, sizeof(float)*size*size);
    	cudaError_t erra;
	erra = cudaMalloc((void**)&d_A, sizeof(float)*size*size);
	if (erra != cudaSuccess) printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(erra));
	cudaMalloc((void**)&d_sd, sizeof(float)*size);
    	cudaMalloc((void**)&d_ic, sizeof(float)*size);
	cudaMemcpy(d_A, A, sizeof(float)*size*size, cudaMemcpyHostToDevice);

	while (err > epsilon)
	{
		
		printf("iteration %d\n",i);
        	cudaMemcpy(d_sd, search_direction, sizeof(float)*size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_ic, intermediate_comp, sizeof(float)*size, cudaMemcpyHostToDevice);
		if (d_A == NULL || d_sd == NULL || d_ic == NULL) {
 		   printf("Device pointers not allocated correctly\n");
		}
		printf("Launching kernel with GridDim=%d, BlockDim=%d\n", GridDim, BlockDim);
		test_access<<<GridDim,BlockDim>>>(d_sd);
		cudaDeviceSynchronize();
		cudaError_t errs = cudaPeekAtLastError();
		if (errs != cudaSuccess)
    			printf("Pre-launch error: %s\n", cudaGetErrorString(errs));		
        	matrix_vector_mult<<<GridDim,BlockDim>>>(size,d_A,d_sd,d_ic);
		
		cudaDeviceSynchronize();
		cudaError_t errc = cudaGetLastError();
		if (errc != cudaSuccess)
			printf("CUDA kernel error: %s\n", cudaGetErrorString(errc));
        	cudaMemcpy(intermediate_comp, d_ic, sizeof(float)*size,cudaMemcpyDeviceToHost);
		
		num = dot_product(size, residual, residual);
		denum = dot_product(size, search_direction, intermediate_comp);
		
		alpha =  num / denum;
	
		vector_add(size,solution,search_direction,alpha,solution);
		
		//vector_copy(size,residual_prev,residual);
		
		vector_add(size,residual,intermediate_comp,alpha,residual);

		denum = num;
		
		num = dot_product(size, residual,residual);

		//denum = dot_product(size, residual_prev, residual_prev);

		beta = num / denum;

		err = sqrt(num);
		
		vector_copy(size,intermediate_comp,residual);
		
		scalar_vector_mult_inplace(size,intermediate_comp,-1);
		
		vector_add(size,intermediate_comp,search_direction,beta,search_direction);

		i++;
	}
	
	printf("Number of iterations: %d\nFinal epsilon: %.12lf\n", i, err);
	free(residual);
	free(residual_prev);
	free(search_direction);
	free(intermediate_comp);
    cudaFree(d_A);
    cudaFree(d_sd);
    cudaFree(d_ic);
	return solution;
}	
