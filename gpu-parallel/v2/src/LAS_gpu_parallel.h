__global__ void matrix_vector_mult(unsigned int size, float* matrix, float* vector, float* vector_storage);
__global__ void test_access(float* vec);
__global__ void dot_product(unsigned int size, float* vector_1, float* vector_2, float* res);
__global__ void vector_add(unsigned int size, float* vector_1, float* vector_2, float* num, float* denum, float* vector_storage);
__global__ void vector_sub(unsigned int size, float* vector_1, float* vector_2, float* num, float* denum, float* vector_storage);
__global__ void vector_copy(unsigned int size, float* vector_out, float* vector_in);
__global__ void compute_residual_gpu(unsigned int size, float* A, float* b, float* x, float* vector_storage, dim3 GridDim, dim3 BlockDim);
__global__ void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha);
void print_vector(unsigned int size, float* vector);
/*
__global__ void dot_product(unsigned int size, float* vector_1, float* vector_2, float* vector_out);
__global__ void vector_add(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
__global__ void vector_sub(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
__global__ void vector_copy(unsigned int size, float* vector_out, float* vector_in);
__global__ void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha);
*/


