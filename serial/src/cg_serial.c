#include"cg_serial.h"
#include"LAS_serial.h"
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define CACHE_BLOCK_SIZE 64 // bytes

// check where to use restrict
double* cg(unsigned int size, double* A, double* b, double* init_g, double epsilon, int debug)
{
	double* solution;
	double* residual;
	double* residual_prev;
	double* search_direction;
	double* intermediate_comp;
	double alpha;
	double beta;
	double num;
	double denum;
	double err;
	int i = 0;
	solution = (double*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(double)*size);
	residual = (double*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(double)*size);
	residual_prev = (double*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(double)*size);
	search_direction = (double*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(double)*size);
	intermediate_comp = (double*)aligned_alloc(CACHE_BLOCK_SIZE, sizeof(double)*size);

	// init vectors to 0
	scalar_vector_mult_inplace(size,solution,0);
	scalar_vector_mult_inplace(size,residual,0);
	scalar_vector_mult_inplace(size,residual_prev,0);
	scalar_vector_mult_inplace(size,search_direction,0);
	scalar_vector_mult_inplace(size,intermediate_comp,0);

	// set current solution to inital guess
	vector_copy(size, solution, init_g);

	// compute residual
	compute_residual(size, A, b, solution, residual);
	
	// determine search direction
	vector_copy(size,search_direction,residual);
	scalar_vector_mult_inplace(size,search_direction,-1);
	
	// compute error measure
	err = dot_product(size, residual, residual);

	err = sqrt(err);
	
	// iterate until error measurement is below epsilon
	while (err > epsilon){
		
		// compute optimal step size alpha
		matrix_vector_mult(size,A,search_direction,intermediate_comp);
		
		num = dot_product(size, residual, residual);

		denum = dot_product(size, search_direction, intermediate_comp);
		
		alpha =  num / denum;
		
		// update solution 
		vector_add(size,solution,search_direction,alpha,solution);
		
		// determine new residual
		vector_copy(size,residual_prev,residual);
		
		vector_add(size,residual,intermediate_comp,alpha,residual);

		// determine optimal beta
		denum = num;
		
		num = dot_product(size, residual,residual);

		//denum = dot_product(size, residual_prev, residual_prev);

		beta = num / denum;
		
		// compute error for next iteration
		err = sqrt(num);
		
		// update search direction
		vector_copy(size,intermediate_comp,residual);
		
		scalar_vector_mult_inplace(size,intermediate_comp,-1);
		
		vector_add(size,intermediate_comp,search_direction,beta,search_direction);
		i++;
	}

	printf("Number of iterations: %d\nFinal epsilon: %.12lf\n", i, sqrt(dot_product(size, residual, residual)));
	free(residual);
	free(residual_prev);
	free(search_direction);
	free(intermediate_comp);
	return solution;
}	
