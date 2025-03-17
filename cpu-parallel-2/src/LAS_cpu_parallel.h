double dot_product(unsigned int size, double* vector_1, double* vector_2);

void matrix_vector_mult(unsigned int size, double* restrict matrix, double* restrict vector, double* restrict vector_storage);

void vector_add(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage);
void vector_sub(unsigned int size, double* vector_1, double* vector_2, double alpha, double* vector_storage);
void vector_copy(unsigned int size, double* vector_out, double* vector_in);
void compute_residual(unsigned int size, double* A, double* b, double* x, double* vector_storage);
void scalar_vector_mult_inplace(unsigned int size,double* vector, double alpha);
void print_vector(unsigned int size, double* vector);
