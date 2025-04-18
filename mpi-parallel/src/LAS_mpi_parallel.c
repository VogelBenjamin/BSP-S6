#include"LAS_serial.h"
#include<stdio.h>
double dot_product(unsigned int size, double* vector_1, double* vector_2)
{
	double storage = 0;
	for (unsigned int i = 0; i < size; ++i)
	{
		storage += vector_1[i]*vector_2[i]; 
	}
	return storage;
}

void matrix_vector_mult(unsigned int size, double* matrix, double* vector, double* vector_storage)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		// use local accumulator
		vector_storage[i] = 0;
		for (unsigned int j = 0; j < size; ++j)
		{
			vector_storage[i] += matrix[i*size+j]*vector[j];
		}
	}
	return;
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
