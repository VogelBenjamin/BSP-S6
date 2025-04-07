#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include"../src/cg_gpu_parallel.h"
#include"../../matrices/matrix_load.h"

#define epsilon 1E-9

int main()
{
    // BCSSTK06
    double* cg_solution;
    double* FFGE_matrix;
    double b_06[420];
    double init_06[420];
    float startTime;
    float endTime;
    float timeElapsed;
    for (unsigned int i = 0; i < 420; ++i)
    {
        b_06[i] = 1;
        init_06[i] = 1;
    }
    FFGE_matrix = load_FFGE("../../matrices/matrix_data/bcsstk06.mtx");
    printf("Start BCSSTK06\n");
    startTime = (float)clock()/CLOCKS_PER_SEC;
    cg_solution = cg(420,FFGE_matrix,b_06,init_06,epsilon,0);
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("Time elapsed: %f\n", timeElapsed);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK06\n");
}