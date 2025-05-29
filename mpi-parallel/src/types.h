#ifndef TYPES_H
#define TYPES_H

typedef struct {
    int M,N,nz,r_start,r_stop,local_nz;
    int* off;
    int* col;
    double* val;
} CSR_MAT;

typedef struct 
{
    int* ranks;
    int* r_start, *r_stop;
} ProcessData;
#endif