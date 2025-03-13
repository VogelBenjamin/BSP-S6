#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include"../src/LAS_serial.h"

#define epsilon 1E-8

int main()
{
    double N = 5;
    double vector_1[5] = {1.0,1.0,1.0,1.0,1.0};
    double vector_2[5] = {1.0,2.0,3.0,4.0,5.0};
    double storage[5];
    double matrix[25] = {
        4.0, 1.0, 0.0, 0.0, 0.0,  // Row 1
        0.0, 4.0, 1.0, 0.0, 0.0,  // Row 2
        0.0, 0.0, 4.0, 1.0, 0.0,  // Row 3
        0.0, 0.0, 0.0, 4.0, 1.0,  // Row 4
        1.0, 0.0, 0.0, 0.0, 4.0   // Row 5
    };

    // dot product test
    assert(dot_product(N,vector_1,vector_2) - 15.0 < epsilon);

    // matrix-vector mult test
    matrix_vector_mult(N,matrix,vector_1,storage);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(storage[i] - 5 < epsilon);
    }

    // vector_add test
    double add_solution[5] = {3.0,5.0,7.0,9.0,11.0};
    vector_add(N,vector_1,vector_2,2,storage);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(storage[i] - add_solution[i] < epsilon);
    }

    // vector_sub test
    double sub_solution[5] = {-1.0,-3.0,-5.0,-7.0,-9.0};
    vector_sub(N,vector_1,vector_2,2,storage);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(storage[i] - sub_solution[i] < epsilon);
    }

    // vector_copy test
    vector_copy(N,storage,vector_2);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(storage[i] - vector_2[i] < epsilon);
    }

    // compute residual
    double res_solution[5] = {5.0,10.0,15.0,20.0,20.0};
    compute_residual(N,matrix,vector_1,vector_2,storage);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(storage[i] - res_solution[i] < epsilon);
    }

    // scalar-vector mult test
    vector_copy(N, storage, vector_1);
    scalar_vector_mult_inplace(N,storage,3);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(storage[i] - 3 < epsilon);
    }

    return 0;
}