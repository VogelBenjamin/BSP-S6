#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include"../src/cg_gpu_parallel.h"
#include"../../../matrices/matrix_load_gpu.h"

#define epsilon 1E-9

int main()
{
    // BCSSTK13
    float* cg_solution;
    float* FFGE_matrix;
    float b_13[2003];
    float init_13[2003];
    float startTime;
    float endTime;
    float timeElapsed;
    for (unsigned int i = 0; i < 2003; ++i)
    {
        b_13[i] = 1;
        init_13[i] = 1;
    }
    FFGE_matrix = load_FFGE_float("../../../matrices/matrix_data/bcsstk13.mtx");
    printf("Start BCSSTK13\n");
    startTime = (float)clock()/CLOCKS_PER_SEC;
    cg_solution = cg(2003,FFGE_matrix,b_13,init_13,epsilon,3);
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("Time elapsed: %f\n", timeElapsed);
    free(FFGE_matrix);
    free(cg_solution);
    printf("Finished BCSSTK13\n");
    return 0;
}
