#include"cg_mpi.h"
#include"LAS_mpi.h"
#include"types.h"
#include<mpi.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define CACHE_BLOCK_SIZE 64 // bytes

// check where to use restrict
double* cg(unsigned int size, CSR_MAT* A, double* b, double* init_g, double epsilon, int debug,int rank, int comm_size,ProcessData* pb)
{
	double* solution;
	double* residual;
	double* residual_prev;
	double* search_direction;
	double* intermediate_comp;
	double* sub_mv_mult_res;
	double alpha;
	double beta;
	double num;
	double denum;
	double err;
	int i = 0;


	solution = (double*)malloc( sizeof(double)*size);
	residual = (double*)malloc( sizeof(double)*size);
	residual_prev = (double*)malloc( sizeof(double)*size);
	search_direction = (double*)malloc( sizeof(double)*size);
	intermediate_comp = (double*)malloc( sizeof(double)*size);
	sub_mv_mult_res = (double*)malloc( sizeof(double)*size);

	// init vectors to 0
	scalar_vector_mult_inplace(size,solution,0);
	scalar_vector_mult_inplace(size,residual,0);
	scalar_vector_mult_inplace(size,residual_prev,0);
	scalar_vector_mult_inplace(size,search_direction,0);
	scalar_vector_mult_inplace(size,intermediate_comp,0);
	scalar_vector_mult_inplace(size,sub_mv_mult_res,0);

	// set current solution to inital guess
	vector_copy(size, solution, init_g);

	// compute residual
	
	MPI_Barrier(MPI_COMM_WORLD);

	matrix_vector_mult(size,A,solution,residual,rank,comm_size,pb);
	
	MPI_Barrier(MPI_COMM_WORLD);

	vector_sub(size,residual,b,1.0,residual);
	
	
	// determine search direction
	vector_copy(size,search_direction,residual);
	scalar_vector_mult_inplace(size,search_direction,-1);
	MPI_Barrier(MPI_COMM_WORLD);

	// compute error measure
	err = dot_product(size, residual, residual);

	err = sqrt(err);
	/*
	if (rank == 0)
	  printf("First error: %lf\n",err);
	*/
	// iterate until error measurement is below epsilon
	
	while (err > epsilon){
		
		// compute optimal step size alpha
		MPI_Barrier(MPI_COMM_WORLD);
		matrix_vector_mult(size,A,search_direction,intermediate_comp, rank, comm_size,pb);
		MPI_Barrier(MPI_COMM_WORLD);
		num = dot_product(size, residual, residual);

		denum = dot_product(size, search_direction, intermediate_comp);
		
		alpha =  num / denum;
		
		// update solution 
		vector_add(size,solution,search_direction,alpha,solution);
		
		// determine new residual
		//vector_copy(size,residual_prev,residual);
		
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
	if (rank == 0)
		printf("Number of iterations: %d\nFinal epsilon: %.12lf\n", i, sqrt(dot_product(size, residual, residual)));
		//print_vector(size,solution);
	free(residual);
	free(residual_prev);
	free(search_direction);
	free(intermediate_comp);
	return solution;
}	
