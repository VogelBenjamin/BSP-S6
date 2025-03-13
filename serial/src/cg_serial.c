#include"cg_serial.h"
#include"LAS_serial.h"
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
double* cg(unsigned int size, double* A, double* b, double* init_g, double epsilon, int debug)
{
	double* solution;
	double* residual;
	double* residual_prev;
	double* search_direction;
	double* intermediate_comp;
	double alpha;
	double beta;
	solution = (double*)malloc(sizeof(double)*size);
	residual = (double*)malloc(sizeof(double)*size);
	residual_prev = (double*)malloc(sizeof(double)*size);
	search_direction = (double*)malloc(sizeof(double)*size);
	intermediate_comp = (double*)malloc(sizeof(double)*size);

	vector_copy(size, solution, init_g);
	compute_residual(size, A, b, solution, residual);
	vector_copy(size,search_direction,residual);
	scalar_vector_mult_inplace(size,search_direction,-1);

	int i = 0;

	while (sqrt(dot_product(size, residual, residual)) > epsilon){
		if(debug == 3 && i > size)
		{
			printf("Unable to converge");
			break;
		}
		if(debug == 2){
			printf("Current epsilon: %lf \n",sqrt(dot_product(size, residual, residual)));
		}
		matrix_vector_mult(size,A,search_direction,intermediate_comp);
		alpha = dot_product(size, residual, residual) / dot_product(size, search_direction, intermediate_comp);

		if(debug == 2){
			printf("Current alpha: %lf, %lf, %.20lf \n",dot_product(size, residual, residual), dot_product(size, search_direction, intermediate_comp), dot_product(size, residual, residual) / dot_product(size, search_direction, intermediate_comp));
		}

		vector_add(size,solution,search_direction,alpha,solution);
		if(debug == 2){
			printf("Current solution: ");
			print_vector(size,solution);
		}
		

		vector_copy(size,residual_prev,residual);
		vector_add(size,residual_prev,intermediate_comp,alpha,residual);
		if(debug == 2){
			printf("Current residual: ");
			print_vector(size,residual);
		}
		


		beta = dot_product(size, residual,residual) / dot_product(size, residual_prev, residual_prev);
		if(debug == 2){
			printf("Current beta: %lf \n",beta);
		}
		vector_copy(size,intermediate_comp,residual);
		scalar_vector_mult_inplace(size,intermediate_comp,-1);
		vector_add(size,intermediate_comp,search_direction,beta,search_direction);
		if(debug == 2){
			printf("Current search dir: ");
			print_vector(size,search_direction);
		}
		
		i++;
	}

	printf("Number of iterations: %d\nFinal epsilon: %.12lf\n", i, sqrt(dot_product(size, residual, residual)));
	free(residual);
	free(residual_prev);
	free(search_direction);
	free(intermediate_comp);
	return solution;
}	
