#include"cg_serial.h"
#define epsilon 1E-8
int main()
{
	double N = 5;
    double vector_1[5] = {1.0,1.0,1.0,1.0,1.0};
    double vector_2[5] = {1.0,2.0,3.0,4.0,5.0};
    double solution[5] = {0.2,0.2,0.2,0.2,0.2};
	double* cg_solution;
    double matrix[25] = {
        4.0, 1.0, 0.0, 0.0, 0.0,  // Row 1
        0.0, 4.0, 1.0, 0.0, 0.0,  // Row 2
        0.0, 0.0, 4.0, 1.0, 0.0,  // Row 3
        0.0, 0.0, 0.0, 4.0, 1.0,  // Row 4
        1.0, 0.0, 0.0, 0.0, 4.0   // Row 5
    };

	cg_solution = cg(N,matrix,vecotr_1,vector_2,epsilon)

	free(cg_solution);

	return 0;
}
