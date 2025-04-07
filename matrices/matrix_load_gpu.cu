#include <stdio.h>
#include <stdlib.h>
#include "mmio_gpu.h"
#include "matrix_load_gpu.h"

float* load_FFGE_float(char* path)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    double* val;

    
    if ((f = fopen(path, "r")) == NULL) 
    {    
        printf("File Not Found\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        int scan_res2 = fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/
    printf("%d %d %d \n",M,N,nz);
    float* return_matrix = (float*)aligned_alloc(64,sizeof(float)*M*N);
    for (i=0; i<M ; ++i)
    {
        for (unsigned int j=0; j<N ; ++j)
        {
            return_matrix[i*M+j] = 0.0f;
        }
    }
    for (i=0; i<nz; i++)
    {
        return_matrix[I[i]*M+J[i]] = (float)val[i];
        return_matrix[J[i]*M+I[i]] = (float)val[i];
    }
    free(I);
    free(J);
    free(val);

	return return_matrix;
}