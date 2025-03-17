#include"LAS_cpu_parallel.h"
#include<stdio.h>
#include <omp.h>

#define CACHE_LINE 8  // 8 doubles

double dot_product(unsigned int size, double* vector_1, double* vector_2)
{
	double acc = 0;
	#pragma omp parallel for schedule(static, 8) reduction(+:acc)
	for (unsigned int i = 0; i < size; ++i)
	{
		acc += vector_1[i]*vector_2[i]; 
	}
	return acc;
}

void matrix_vector_mult(unsigned int size, double* restrict matrix, double* restrict vector, double* restrict vector_storage)
{
	#pragma omp parallel for schedule(static, CACHE_LINE)
	for (unsigned int i = 0; i < size; ++i)
	{
		const double* row = &matrix[i * size];
		double acc = 0;
		#pragma omp simd aligned(matrix, vector : 64) reduction(+:acc)
		for (unsigned int j = 0; j < size; ++j)
		{
			acc += row[j]*vector[j];
		}
		vector_storage[i] = acc;
	}
	return;
}

void vector_add(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage)
{
	#pragma omp parallel for schedule(static, CACHE_LINE)
	for (unsigned int i = 0; i < size; ++i)
	{
		vector_storage[i] = vector_1[i] + alpha*vector_2[i];	
	}
	return;
}

void vector_sub(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage)
{
	#pragma omp parallel for schedule(static, CACHE_LINE)
	for (unsigned int i = 0; i < size; ++i)
	{
			vector_storage[i] = vector_1[i] - alpha*vector_2[i];
	}
	return;
}

void vector_copy(unsigned int size, double* vector_out, double* vector_in)
{
	#pragma omp parallel for schedule(static, CACHE_LINE)
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
	#pragma omp parallel for schedule(static, CACHE_LINE)
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
