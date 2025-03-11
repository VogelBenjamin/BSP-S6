#include"cg_serial.h"
#include"LAS_serial.h"
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
double* cg(unsigned int size, double* A, double* b, double* init_g, double epsilon)
{
	double* solution;
	double* residual;
	double* residual_prev;
	double* search_direction;
	double* intermediate_comp;
	double alpha;
	double beta;
	solution = malloc(sizeof(double)*size);
	residual = malloc(sizeof(double)*size);
	residual_prev = malloc(sizeof(double)*size);
	search_direction = malloc(sizeof(double)*size);
	intermediate_comp = malloc(sizeof(double)*size);

	vector_copy(size, solution, init_g);
	compute_residual(size, A, b, solution, residual);
	vector_copy(size,search_direction,residual);
	scalar_vector_mult_inplace(size,search_direction,-1);

	int i = 0;

	while (sqrt(dot_product(size, residual, residual)) > epsilon){
		//printf("Current epsilon: %lf \n",sqrt(dot_product(size, residual, residual)));
		matrix_vector_mult(size,A,search_direction,intermediate_comp);
		alpha = dot_product(size, residual, residual) / dot_product(size, search_direction, intermediate_comp);
		//printf("Current alpha: %lf \n", alpha);

		vector_add(size,solution,search_direction,alpha,solution);
		//printf("Current solution: ");
		//print_vector(size,solution);

		vector_copy(size,residual_prev,residual);
		vector_add(size,residual_prev,intermediate_comp,alpha,residual);
		//printf("Current residual: ");
		//print_vector(size,residual);


		beta = dot_product(size, residual,residual) / dot_product(size, residual_prev, residual_prev);
		//printf("Current beta: %lf \n", beta);

		vector_copy(size,intermediate_comp,residual);
		scalar_vector_mult_inplace(size,intermediate_comp,-1);
		vector_add(size,intermediate_comp,search_direction,beta,search_direction);
		
		//printf("Current search dir: ");
		//print_vector(size,search_direction);
		i++;
	}

	
	free(residual);
	free(residual_prev);
	free(search_direction);
	free(intermediate_comp);
	return solution;
}	
