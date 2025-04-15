#ifndef CG_GPU_PARALLEL_H
#define CG_GPU_PARALLEL_H

#ifdef __cplusplus
extern "C" {
#endif

float* cg(unsigned int size, float* A, float* b, float* init_g, float epsilon, int debug);

#ifdef __cplusplus
}
#endif

#endif

