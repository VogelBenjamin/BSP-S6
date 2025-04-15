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
float* cg(unsigned int size, float* A, float* b, float* init_g, float epsilon, int debug)
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

	// allocate host data
	solution = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	residual = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	residual_prev = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	search_direction = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);
	intermediate_comp = (float*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(float)*size);

	// allocate device data
	float* d_A, *d_b, *d_sd, *d_ic, *d_sol, *d_res, *d_resp;
	cudaMalloc((void**)&d_A, sizeof(float)*size*size);
	cudaMalloc((void**)&d_b, sizeof(float)*size);
	cudaMalloc((void**)&d_sd, sizeof(float)*size);
	cudaMalloc((void**)&d_ic, sizeof(float)*size);
	cudaMalloc((void**)&d_sol, sizeof(float)*size);
	cudaMalloc((void**)&d_res, sizeof(float)*size);
	cudaMalloc((void**)&d_resp, sizeof(float)*size);


	// initialise host data
	scalar_vector_mult_inplace(size,solution,0);
	scalar_vector_mult_inplace(size,residual,0);
	scalar_vector_mult_inplace(size,residual_prev,0);
	scalar_vector_mult_inplace(size,search_direction,0);
	scalar_vector_mult_inplace(size,intermediate_comp,0);

	vector_copy(size, solution, init_g);
	
	// initialise device data
	cudaMemcpy(d_A, A, sizeof(float)*size*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sd, solution, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ic, intermediate_comp, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, residual, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_resp, residual_prev, sizeof(float)*size, cudaMemcpyHostToDevice);

	// residual calc
	// A = d_A , d_sd = solution , d_ic = residual
	matrix_vector_mult<<<GridDim,BlockDim>>>(size,d_A,d_sd,d_res);
	cudaDeviceSynchronize();
	vector_sub<<<GridDim,BlockDim>>>(size,d_res,d_b,1.0,d_res);
	cudaDeviceSynchronize();
	vector_copy<<<GridDim,BlockDim>>>(size,d_sd,d_res);
	cudaDeviceSynchronize();
	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_sd,-1);
	cudaDeviceSynchronize();

	cudaMemcpy(residual,d_res,sizeof(float)*size,cudaMemcpyDeviceToHost);

	err = dot_product(size, residual, residual);

	err = sqrt(err);

	

	while (err > epsilon)
	{
		
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
		
		cudaMemcpy(residual,d_res,sizeof(float)*size,cudaMemcpyDeviceToHost);
		cudaMemcpy(intermediate_comp,d_ic,sizeof(float)*size,cudaMemcpyDeviceToHost);
		num = dot_product(size, residual, residual);
		denum = dot_product(size, search_direction, intermediate_comp);
		
		alpha =  num / denum;
	
		vector_add<<<GridDim,BlockDim>>>(size,d_sol,d_sd,alpha,d_sol);
		cudaDeviceSynchronize();
		//vector_copy(size,residual_prev,residual);
		
		vector_add<<<GridDim,BlockDim>>>(size,d_res,d_ic,alpha,d_res);
		cudaDeviceSynchronize();
		denum = num;
		
		cudaMemcpy(residual,d_res,sizeof(float)*size,cudaMemcpyDeviceToHost);
		num = dot_product(size, residual,residual);

		//denum = dot_product(size, residual_prev, residual_prev);

		beta = num / denum;

		err = sqrt(num);
		
		vector_copy<<<GridDim,BlockDim>>>(size,d_ic,d_res);
		
		scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_ic,-1);
		
		vector_add<<<GridDim,BlockDim>>>(size,d_ic,d_sd,beta,d_sd);

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
	cudaFree(d_b);
    cudaFree(d_res);
    cudaFree(d_resp);
	cudaFree(d_sol);
	return solution;
}	
