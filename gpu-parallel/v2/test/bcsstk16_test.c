#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include"../src/cg_gpu_parallel.h"
#include"../../../matrices/matrix_load_gpu.h"

#define epsilon 1E-9

int main()
{
    // BCSSTK16
    float* cg_solution;
    float* FFGE_matrix;
    float b_16[4884];
    float init_16[4884];
    float startTime;
    float endTime;
    float timeElapsed;
    for (unsigned int i = 0; i < 4884; ++i)
    {
        b_16[i] = 1;
        init_16[i] = 1;
    }
    FFGE_matrix = load_FFGE_float("../../../matrices/matrix_data/bcsstk16.mtx");
    printf("Start BCSSTK16\n");
    startTime = (float)clock()/CLOCKS_PER_SEC;
    cg_solution = cg(4884,FFGE_matrix,b_16,init_16,epsilon,3);
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("Time elapsed: %f\n", timeElapsed);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK16\n");
    return 0;
}
