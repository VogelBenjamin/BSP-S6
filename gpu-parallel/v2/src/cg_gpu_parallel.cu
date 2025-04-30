#include"cg_gpu_parallel.h"
#include"LAS_gpu_parallel.h"
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>

#define CACHE_BLOCK_SIZE 64 // bytes
#define BLOCK_DIM 64
// compute step size


// check where to use 
float* cg(unsigned int size, float* A, float* b, float* init_g, float epsilon, int debug)
{
	float*  solution;
	float*  residual;
	float*  residual_prev;
	float*  search_direction;
	float*  intermediate_comp;
	float *num = (float*)malloc(sizeof(float));
	float *one = (float*)malloc(sizeof(float));
	float err;
	int i = 0;
	*one = 1.0f;
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

	float*d_num, *d_denum, *d_one;
	cudaMalloc((void**)&d_num, sizeof(float));
	cudaMalloc((void**)&d_denum, sizeof(float));
	cudaMalloc((void**)&d_one, sizeof(float));

	cudaMemcpy(d_one,one,sizeof(float),cudaMemcpyHostToDevice);

	// initialise device data
	
	cudaError_t errs_init;
	
	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_sol,0);
	
	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_res,0);
	
	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_resp,0);
	
	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_sd,0);
	
	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_ic,0);

	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(1,d_num,0);

	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(1,d_denum,0);
	
	cudaMemcpy(d_A, A, sizeof(float)*size*size, cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_sol, init_g, sizeof(float)*size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_b, b, sizeof(float)*size, cudaMemcpyHostToDevice);

	errs_init = cudaPeekAtLastError();
	if (errs_init != cudaSuccess)
		printf("Pre-launch error init 8: %s\n", cudaGetErrorString(errs_init));
	
	
	// residual calc
	// A = d_A , d_sol = solution , d_ic = residual
	cudaError_t errs_ferr;

	matrix_vector_mult<<<GridDim,BlockDim>>>(size,d_A,d_sol,d_res);

	cudaDeviceSynchronize();

	vector_sub<<<GridDim,BlockDim>>>(size,d_res,d_b,d_one,d_one,d_res);

	cudaDeviceSynchronize();

	vector_copy<<<GridDim,BlockDim>>>(size,d_sd,d_res);
	
	cudaDeviceSynchronize();

	scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_sd,-1);

	
	errs_ferr = cudaPeekAtLastError();	
	if (errs_ferr != cudaSuccess)
		printf("Pre-launch error fmvm 4: %s\n", cudaGetErrorString(errs_ferr));

	cudaDeviceSynchronize();

	dot_product<<<GridDim,BlockDim>>>(size, d_res, d_res, d_num);
	cudaMemcpy(num,d_num,sizeof(float),cudaMemcpyDeviceToHost);
	err = sqrt(*num);

	printf("Launching kernel with GridDim=%d, BlockDim=%d\n", GridDim, BlockDim);
	
	while (err > epsilon)
	{
		cudaMemset(d_num, 0, sizeof(float));

		cudaMemset(d_denum, 0, sizeof(float));
		
		//printf("Error: %f\n",err);	
		cudaError_t errc;
		// compute alpha
        matrix_vector_mult<<<GridDim,BlockDim>>>(size,d_A,d_sd,d_ic);
		
		dot_product<<<GridDim,BlockDim>>>(size, d_res, d_res, d_num);

		cudaDeviceSynchronize();
		
		dot_product<<<GridDim,BlockDim>>>(size, d_sd, d_ic, d_denum);
		errc = cudaGetLastError();
		if (errc != cudaSuccess)
			printf("CUDA kernel error: %s\n", cudaGetErrorString(errc));
		cudaDeviceSynchronize();

		// update residual and search direction
		vector_add<<<GridDim,BlockDim>>>(size,d_sol,d_sd,d_num,d_denum,d_sol);
		
		vector_add<<<GridDim,BlockDim>>>(size,d_res,d_ic,d_num,d_denum,d_res);
		errc = cudaGetLastError();
		if (errc != cudaSuccess)
			printf("CUDA kernel error: %s\n", cudaGetErrorString(errc));

		cudaDeviceSynchronize();

		// compute beta
		vector_copy<<<GridDim,BlockDim>>>(1,d_denum, d_num);
		
		cudaMemset(d_num, 0, sizeof(float)); // already synchronizes device

		dot_product<<<GridDim,BlockDim>>>(size, d_res,d_res,d_num);

		cudaMemcpy(num,d_num,sizeof(float),cudaMemcpyDeviceToHost); // synchronizes
		err = sqrt(*num);
		
		vector_copy<<<GridDim,BlockDim>>>(size,d_ic,d_res);
		
		cudaDeviceSynchronize();

		scalar_vector_mult_inplace<<<GridDim,BlockDim>>>(size,d_ic,-1);

		cudaDeviceSynchronize();

		vector_add<<<GridDim,BlockDim>>>(size,d_ic,d_sd,d_num,d_denum,d_sd);

		errc = cudaGetLastError();
		if (errc != cudaSuccess)
			printf("CUDA kernel error: %s\n", cudaGetErrorString(errc));
		
		
		
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
