__global__ void matrix_vector_mult(unsigned int size, float* matrix, float* vector, float* result);
__global__ void test_access(float* vec);
float dot_product(unsigned int size, float* vector_1, float* vector_2);
void vector_add(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
void vector_sub(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
void vector_copy(unsigned int size, float* vector_out, float* vector_in);
void compute_residual_gpu(unsigned int size, float* A, float* b, float* x, float* vector_storage, dim3 GridDim, dim3 BlockDim);
void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha);
void print_vector(unsigned int size, float* vector);
/*
__global__ void dot_product(unsigned int size, float* vector_1, float* vector_2, float* vector_out);
__global__ void vector_add(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
__global__ void vector_sub(unsigned int size, float* vector_1, float* vector_2, float alpha, float* vector_storage);
__global__ void vector_copy(unsigned int size, float* vector_out, float* vector_in);
__global__ void scalar_vector_mult_inplace(unsigned int size,float* vector, float alpha);
*/


