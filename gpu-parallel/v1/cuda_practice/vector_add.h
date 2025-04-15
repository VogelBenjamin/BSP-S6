#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

__global__ void vector_add(float *a, float *b, float *out, int N);
void exec();
#endif // VECTOR_ADD_H
