#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include"../src/cg_serial.h"
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

    // BCSSTK06
    double b[420];
    double init[420];
    for (unsigned int i = 0; i < 420; ++i)
    {
        b[i] = 1;
        init[i] = 1;
    }
    double* FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk06.mtx");
    printf("This worked\n");
    printf("Start BCSSTK06\n");
    cg_solution = cg(420,FFGE_matrix,b,init,epsilon,0);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK06\n");

    // BCSSTK13
    double b[2003];
    double init[2003];
    for (unsigned int i = 0; i < 2003; ++i)
    {
        b[i] = 1;
        init[i] = 1;
    }
    double* FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk13.mtx");
    printf("This worked\n");
    printf("Start BCSSTK13\n");
    cg_solution = cg(2003,FFGE_matrix,b,init,epsilon,3);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK13\n");

    // BCSSTK16
    double b[4884];
    double init[4884];
    for (unsigned int i = 0; i < 4884; ++i)
    {
        b[i] = 1;
        init[i] = 1;
    }
    double* FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk16.mtx");
    printf("This worked\n");
    printf("Start BCSSTK16\n");
    cg_solution = cg(4884,FFGE_matrix,b,init,epsilon,3);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK16\n");

    // BCSSTK17
    double b[10974];
    double init[10974];
    for (unsigned int i = 0; i < 10974; ++i)
    {
        b[i] = 1;
        init[i] = 1;
    }
    double* FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk17.mtx");
    printf("This worked\n");
    printf("Start BCSSTK17\n");
    cg_solution = cg(10974,FFGE_matrix,b,init,epsilon,3);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK17\n");
    return 0;
}

