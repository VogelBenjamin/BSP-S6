#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmio.h"

int main(int argc, char* argv[]) {
    FILE *f_in, *f_out;
    int M, N, nz;
    int ret_code;
    MM_typecode matcode;
    double val;
    int I, J;

    double* matrix;
    double *nz_val;
    int *col_idx, *os_idx; 

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
        return 1;
    }

    // Open input file
    f_in = fopen(argv[1], "r");
    if (!f_in) {
        perror("Error opening input file");
        return 1;
    }

    // Read Matrix Market banner
    if (mm_read_banner(f_in, &matcode) != 0) {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        fclose(f_in);
        return 1;
    }

    // Check matrix type
    if (mm_is_complex(matcode)) {
        fprintf(stderr, "Complex matrices are not supported.\n");
        fclose(f_in);
        return 1;
    }

    // Read matrix dimensions
    if ((ret_code = mm_read_mtx_crd_size(f_in, &M, &N, &nz)) != 0) {
        fprintf(stderr, "Error reading matrix size\n");
        fclose(f_in);
        return 1;
    }

    printf("Matrix size: %d x %d with %d non-zeros\n", M, N, nz);

    // Allocate memory
    matrix = (double*)calloc(M * N, sizeof(double));  // Use calloc to initialize to 0

    if (!matrix || !nz_val || !col_idx || !os_idx) {
        perror("Memory allocation failed");
        if (matrix) free(matrix);
        if (nz_val) free(nz_val);
        if (col_idx) free(col_idx);
        if (os_idx) free(os_idx);
        fclose(f_in);
        return 1;
    }

    // Read matrix entries
    for (int i = 0; i < nz; i++) {
        if (fscanf(f_in, "%d %d %lf\n", &I, &J, &val) != 3) {
            fprintf(stderr, "Error reading entry %d\n", i + 1);
            free(matrix);
            free(nz_val);
            free(col_idx);
            free(os_idx);
            fclose(f_in);
            return 1;
        }
        I--;  // Convert to 0-based
        J--;
        matrix[I * N + J] = val;  // Note: N columns, not M
        //If symmetric, you might want to set the symmetric entry too
        matrix[J * N + I] = val;
    }
    fclose(f_in);

    nz *= 2; // store both upper and lower triangualr matrix
    nz -= M;
    nz_val = (double*)malloc(nz * sizeof(double));
    col_idx = (int*)malloc(nz * sizeof(int));
    os_idx = (int*)malloc((M + 1) * sizeof(int));
    // Build CSR format
    int nz_cnt = 0;
    os_idx[0] = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (matrix[i * N + j] != 0) {
                if (nz_cnt >= nz) {
                    fprintf(stderr, "Error: Found more non-zeros than expected\n");
                    free(matrix);
                    free(nz_val);
                    free(col_idx);
                    free(os_idx);
                    return 1;
                }
                nz_val[nz_cnt] = matrix[i * N + j];
                col_idx[nz_cnt] = j;
                nz_cnt++;
            }
        }
        os_idx[i + 1] = nz_cnt;
    }
    printf("Matrix last val %lf\n", matrix[M*N-1]);
    printf("Matrix last val v2 %lf, %d, %d\n", nz_val[nz_cnt-1], nz_cnt,nz);
    // Write binary file
    f_out = fopen("matrix_data/binary_bcsstk17.bin", "wb");
    if (!f_out) {
        perror("Error creating output file");
        free(matrix);
        free(nz_val);
        free(col_idx);
        free(os_idx);
        return 1;
    }

    fwrite(&M, sizeof(int), 1, f_out);
    fwrite(&N, sizeof(int), 1, f_out);
    fwrite(&nz, sizeof(int), 1, f_out);
    fwrite(os_idx, sizeof(int), M + 1, f_out);
    fwrite(col_idx, sizeof(int), nz, f_out);
    fwrite(nz_val, sizeof(double), nz, f_out);

    // Cleanup
    fclose(f_out);
    free(matrix);
    free(nz_val);
    free(col_idx);
    free(os_idx);

    return 0;
}