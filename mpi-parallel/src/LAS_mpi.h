#include"types.h"

double dot_product(unsigned int size, double* vector_1, double* vector_2);

void matrix_vector_mult(unsigned int size, CSR_MAT* matrix, double* vector, double* vector_storage, int rank, int comm_size,ProcessData* pb);

void vector_add(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage);
void vector_sub(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage);
void vector_copy(unsigned int size, double* vector_out, double* vector_in);
void scalar_vector_mult_inplace(unsigned int size,double* vector, double alpha);
void print_vector(unsigned int size, double* vector);
