#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include"../src/cg_cpu_parallel.h"
#include"../../matrices/matrix_load.h"

#define epsilon 1E-9

int main()
{
    double N = 5;
    double vector_1[5] = {1.0,1.0,1.0,1.0,1.0};
    double vector_2[5] = {1.0,2.0,3.0,4.0,5.0};
    double storage[5];
    
    // simple-matrix cg test
    double cg_matrix[25] = {
        5.0, 1.0, 0.0, 0.0, 1.0,  
        1.0, 5.0, 1.0, 0.0, 0.0,  
        0.0, 1.0, 5.0, 1.0, 0.0,  
        0.0, 0.0, 1.0, 5.0, 1.0,  
        1.0, 0.0, 0.0, 1.0, 5.0   
    };
    double* cg_solution;
    printf("Start small test problem\n");
    cg_solution = cg(N,cg_matrix,vector_1,vector_2,epsilon,0);
    for (unsigned int i = 0; i < N; ++i)
    {
        assert(cg_solution[i] - (1.0/7.0) < epsilon);
    }
    free(cg_solution);
    printf("Finished small test problem\n");

    double* FFGE_matrix;
    // BCSSTK06
    double b_06[420];
    double init_06[420];
    for (unsigned int i = 0; i < 420; ++i)
    {
        b_06[i] = 1;
        init_06[i] = 1;
    }
    FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk06.mtx");
    printf("Start BCSSTK06\n");
    cg_solution = cg(420,FFGE_matrix,b_06,init_06,epsilon,0);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK06\n");

    // BCSSTK13
    /*
    double b_13[2003];
    double init_13[2003];
    for (unsigned int i = 0; i < 2003; ++i)
    {
        b_13[i] = 1;
        init_13[i] = 1;
    }
    FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk13.mtx");
    printf("Start BCSSTK13\n");
    cg_solution = cg(2003,FFGE_matrix,b_13,init_13,epsilon,3);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK13\n");
    */
    // BCSSTK16
    double b_16[4884];
    double init_16[4884];
    for (unsigned int i = 0; i < 4884; ++i)
    {
        b_16[i] = 1;
        init_16[i] = 1;
    }
    FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk16.mtx");
    printf("Start BCSSTK16\n");
    cg_solution = cg(4884,FFGE_matrix,b_16,init_16,epsilon,0);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK16\n");

    // BCSSTK17
    /*
    double b_17[10974];
    double init_17[10974];
    for (unsigned int i = 0; i < 10974; ++i)
    {
        b_17[i] = 1;
        init_17[i] = 1;
    }
    FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk17.mtx");
    printf("Start BCSSTK17\n");
    cg_solution = cg(10974,FFGE_matrix,b_17,init_17,epsilon,0);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK17\n");
    */
    return 0;
}

