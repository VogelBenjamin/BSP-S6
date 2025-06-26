#include"LAS_mpi.h"
#include"cg_mpi.h"
#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
double dot_product(unsigned int size, double* vector_1, double* vector_2)
{
	double storage = 0;
	for (unsigned int i = 0; i < size; ++i)
	{
		storage += vector_1[i]*vector_2[i]; 
	}
	return storage;
}

void matrix_vector_mult_compute(unsigned int size, CSR_MAT* matrix, double* vector, double* vector_storage, int rank, int comm_size)
{
	int start = matrix->r_start;
	int stop = matrix->r_stop;
	int diff = stop-start;
	int start_idx = 0;
	int end_idx = 0;
	int col;
	double val;

	for (int i = 0 ; i < diff; ++i)
	{
		vector_storage[i] = 0;
	}
	for (unsigned int i = 0; i < diff; ++i)
	{
		// treat only relevant values
		start_idx = matrix->off[i];
		end_idx = matrix->off[i+1];
		for (int j = start_idx; j < end_idx; j++)
		{
			col = matrix->col[j];
			val = matrix->val[j];
			vector_storage[i] += val*vector[col];
		}
		
	}
	
	return;
}


void matrix_vector_mult_gather(unsigned int size, CSR_MAT* matrix, double* vector, double* vector_storage, int rank, int comm_size, ProcessData* pb)
{
	
	int *rcv_cnt = (int*)malloc((comm_size)*sizeof(int));
	int *displs = (int*)malloc((comm_size)*sizeof(int));
	int curr_step = 0;
	int send_size = pb->r_stop[rank] - pb->r_start[rank];
	int diff;
	for (int i = 0; i < comm_size; i++)
	{
		diff = pb->r_stop[i] - pb->r_start[i];
		rcv_cnt[i] = diff;
		displs[i] = curr_step;
		curr_step += diff;
	}
	//printf("rank: %d, send_size: %d\n", rank,send_size);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allgatherv(vector,send_size,MPI_DOUBLE,vector_storage,rcv_cnt,displs,MPI_DOUBLE,MPI_COMM_WORLD);

	free(rcv_cnt);
	free(displs);
}

void matrix_vector_mult(unsigned int size, CSR_MAT* matrix, double* vector, double* vector_storage, int rank, int comm_size,ProcessData* pb)
{

	int row_cnt = pb->r_stop[rank]-pb->r_start[rank];
	double* tmp_storage = (double*)malloc(row_cnt*sizeof(double));

	MPI_Barrier(MPI_COMM_WORLD);
	
	matrix_vector_mult_compute(size,matrix,vector,tmp_storage,rank,comm_size);
	
	MPI_Barrier(MPI_COMM_WORLD);

	matrix_vector_mult_gather(size,matrix,tmp_storage,vector_storage,rank,comm_size,pb);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
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
