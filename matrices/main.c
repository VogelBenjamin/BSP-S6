#include"matrix_load.h"
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include <math.h>

#define epsilon 1e-9
int check_symmetry(unsigned int size, double* matrix)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        for (unsigned int j = 0; j <= i; ++j)
        {
            if(matrix[size*i+j] - matrix[size*j+i] > epsilon)
            {
                printf("Not symmetric:\nmatrix[%d][%d] = %lf \nmatrix[%d][%d] = %lf\n", i ,j, matrix[size*i+j], j,i,matrix[size*j+i]);
                return 0;
            }
        }
        
    } 
    return 1;
}

// The following function has been written by AI
/*
Prompt used:
write me C code that takes the size of an nxn matrix and the matrix itself, 
solved in a contiguous matrix, i.e. access its values m[i*n+j]. 
the function should return a int (0 or 1) depending on whether the matrix is positive definite
*/
int is_positive_definite(int n, double *matrix) {
    const double eps = 1e-8;  // Tolerance for floating point comparisons
    
    // Check if matrix is square and has valid size
    if (n <= 0 || matrix == NULL) return 0;

    // Check symmetry (matrix must be symmetric for positive definiteness)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(matrix[i*n + j] - matrix[j*n + i]) > eps) {
                return 0;  // Not symmetric
            }
        }
    }

    // Allocate memory for Cholesky decomposition
    double *L = (double *)calloc(n * n, sizeof(double));
    if (!L) return 0;  // Memory allocation failed

    // Perform Cholesky decomposition
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            
            // Calculate sum of L[i][k] * L[j][k]
            for (int k = 0; k < j; k++) {
                sum += L[i*n + k] * L[j*n + k];
            }

            // Diagonal elements
            if (i == j) {
                double a_ii = matrix[i*n + i];
                double diff = a_ii - sum;
                if (diff <= eps) {  // Check for positive definiteness
                    free(L);
                    return 0;
                }
                L[i*n + i] = sqrt(diff);
            }
            // Off-diagonal elements
            else {
                double a_ij = matrix[i*n + j];
                double diff = a_ij - sum;
                double l_jj = L[j*n + j];
                
                if (l_jj <= eps) {  // Avoid division by near-zero
                    free(L);
                    return 0;
                }
                L[i*n + j] = diff / l_jj;
            }
        }
    }

    free(L);
    return 1;  // Cholesky decomposition succeeded
}

int main()
{
    double* matrix = load_FFGE("matrix_data/bcsstk06.mtx");
    int N = 420;
    printf("Loaded matrix successfully\n");
    if(check_symmetry(N,matrix))
    {
        printf("matrix is symmetric!\n");
    }
    if(is_positive_definite(N,matrix))
    {
        printf("matrix is positive defininte!\n");
    }
    return 0;
}