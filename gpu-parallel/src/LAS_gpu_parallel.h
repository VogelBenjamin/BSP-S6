__global__ void matrix_vector_mult(unsigned int size, double* matrix, double* vector, double* vector_storage);
double dot_product(unsigned int size, double* vector_1, double* vector_2);
void vector_add(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage);
void vector_sub(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage);
void vector_copy(unsigned int size, double* vector_out, double* vector_in);
void compute_residual(unsigned int size, double* A, double* b, double* x, double* vector_storage);
void scalar_vector_mult_inplace(unsigned int size,double* vector, double alpha);
void print_vector(unsigned int size, double* vector);
/*
__global__ void dot_product(unsigned int size, float* vector_1, float* vector_2, float* vector_out);
__global__ void vector_add(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
__global__ void vector_sub(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
__global__ void vector_copy(unsigned int size, float* vector_out, float* vector_in);
__global__ void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha);
*/


