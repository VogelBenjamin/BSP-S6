#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include"../src/cg_serial.h"
#include"../../matrices/matrix_load.h"

#define epsilon 1E-9

int main()
{
    // BCSSTK17
    double* cg_solution;
    double* FFGE_matrix;
    double b_17[10974];
    double init_17[10974];
    float startTime;
    float endTime;
    float timeElapsed;
    for (unsigned int i = 0; i < 10974; ++i)
    {
        b_17[i] = 1;
        init_17[i] = 1;
    }
    FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk17.mtx");
    printf("Start BCSSTK17\n");
    startTime = (float)clock()/CLOCKS_PER_SEC;
    cg_solution = cg(10974,FFGE_matrix,b_17,init_17,epsilon,3);
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("Time elapsed: %f\n", timeElapsed);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK17\n");
    return 0;
}