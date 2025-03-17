#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<omp.h>
#include"../src/LAS_cpu_parallel.h"

#define epsilon 1E-8

int main()
{
    double N = 5;
    double vector_1[5] = {1.0,1.0,1.0,1.0,1.0};
    double vector_2[5] = {1.0,2.0,3.0,4.0,5.0};
    double add_solution[5] = {3.0,5.0,7.0,9.0,11.0};
    double sub_solution[5] = {-1.0,-3.0,-5.0,-7.0,-9.0};
    double res_solution[5] = {5.0,10.0,15.0,20.0,20.0};
    double storage_1[5];
    double storage_2[5];
    double storage_3[5];
    double storage_4[5];
    double storage_5[5];
    double storage_6[5];
    double dot_res = 0;
    double matrix[25] = {
        4.0, 1.0, 0.0, 0.0, 0.0,  // Row 1
        0.0, 4.0, 1.0, 0.0, 0.0,  // Row 2
        0.0, 0.0, 4.0, 1.0, 0.0,  // Row 3
        0.0, 0.0, 0.0, 4.0, 1.0,  // Row 4
        1.0, 0.0, 0.0, 0.0, 4.0   // Row 5
    };

    // dot product test
    #pragma omp parallel reduction(+: dot_res)
    {
        double partial_sum =  dot_product(N,vector_1,vector_2);
        dot_res += partial_sum;
    

        // matrix-vector mult 
        
        matrix_vector_mult(N,matrix,vector_1,storage_1);
        
        
        

        // vector_add test
        
        
        vector_add(N,vector_1,vector_2,2,storage_2);
        
        

        // vector_sub test
        
        
        vector_sub(N,vector_1,vector_2,2,storage_3);
        
        

        // vector_copy test
        
        vector_copy(N,storage_4,vector_2);
        
        // compute residual
        
        compute_residual(N,matrix,vector_1,vector_2,storage_5);
        
        

        // scalar-vector mult test
        vector_copy(N, storage_6, vector_1);
        scalar_vector_mult_inplace(N,storage_6,3);
        
        
    }

    assert(abs(dot_res - 15.0) < epsilon);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(abs(storage_1[i]) - 5 < epsilon);
    }

    for (unsigned int i = 0; i < N; ++i)
    {
        assert(abs(storage_2[i] - add_solution[i]) < epsilon);
    }

    for (unsigned int i = 0; i < N; ++i)
    {
        assert(abs(storage_3[i] - sub_solution[i]) < epsilon);
    }

    for (unsigned int i = 0; i < N; ++i)
    {
        assert(abs(storage_4[i] - vector_2[i]) < epsilon);
    }


    for (unsigned int i = 0; i < N; ++i)
    {
        assert(abs(storage_5[i] - res_solution[i]) < epsilon);
    }

    for (unsigned int i = 0; i < N; ++i)
    {
        assert(abs(storage_6[i] - 3) < epsilon);
    }

    return 0;
}