#include"types.h"

double* cg(unsigned int size, CSR_MAT* A, double* b, double* init_g, double epsilon, int debug,int rank, int comm_size, ProcessData* pb);