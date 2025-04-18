#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<assert.h>
#include"../src/cg_gpu_parallel.h"
#include"../src/LAS_gpu_parallel.h"

#define epsilon 1e-5

int main()
{
    float N = 5;
    float vector_1[5] = {1.0,1.0,1.0,1.0,1.0};
    float vector_2[5] = {1.0,2.0,3.0,4.0,5.0};
    float storage[5];
    
    // simple-matrix cg test
    float cg_matrix[25] = {
        5.0, 1.0, 0.0, 0.0, 1.0,  
        1.0, 5.0, 1.0, 0.0, 0.0,  
        0.0, 1.0, 5.0, 1.0, 0.0,  
        0.0, 0.0, 1.0, 5.0, 1.0,  
        1.0, 0.0, 0.0, 1.0, 5.0   
    };
    float* cg_solution;
    printf("Start small test problem\n");
    cg_solution = cg(N,cg_matrix,vector_1,vector_2,epsilon,0);
    print_vector(5,cg_solution);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(cg_solution[i] - (1.0/7.0) < epsilon);
    }
    free(cg_solution);
    printf("Finished small test problem\n");
}
