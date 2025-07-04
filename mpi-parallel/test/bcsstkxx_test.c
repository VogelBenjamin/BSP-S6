#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<mpi.h>
#include<stdbool.h>
#include<string.h>
#include"../src/cg_mpi.h"

#define epsilon 1E-9



int main(int argc, char *argv[])
{
    // BCSSTKXX
    int rank, size, M, N, nz;
    double* cg_solution;
    CSR_MAT FFGE_submatrix;
    double* b_06;
    double* init_06;
    int submatrix_row_n;
    int submatrix_remainder;
    int local_work;
    int offset;
    int* matrix_offset;
    char *file_path;

    MPI_File fh;
    MPI_Offset byte_offset;
    

    // initialise MPI environemnt
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (rank == 0)
        printf("Single File read start\n");

    if (argc != 2)
    {
        printf("Missing argument\n");
        return 1;
    }
    
    // fetch matrix dimensions and number of elements
    file_path = argv[1];
    if (rank == 0) {
        FILE *fp = fopen(file_path, "rb");
        fread(&M, sizeof(int), 1, fp);
        fread(&N, sizeof(int), 1, fp);
        fread(&nz, sizeof(int), 1, fp);
        fclose(fp);
    }

    if (rank ==0)
        printf("M: %d, N: %d, nz: %d\n",M,N,nz);
    if (rank == 0)
        printf("Single File read end\n");

    // broadcast dimensions and element count among all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // initialise problem
    if (rank == 0)
        printf("Init the constant vectors.\n");

    b_06 = (double*)malloc(sizeof(double)*M);
    init_06 = (double*)malloc(sizeof(double)*M);


    
    for (unsigned int i = 0; i < M; ++i)
    {
        b_06[i] = 1;
        init_06[i] = 1;
    }

    if (rank == 0)
        printf("Init the constant vectors done.\n");
    

    matrix_offset = (int*)malloc(sizeof(int)*(M+1));

    

    if (rank == 0)
        printf("Start Reading Matrix\n");

    MPI_File_open(MPI_COMM_WORLD, file_path,
                  MPI_MODE_RDONLY, MPI_INFO_NULL, 
                  &fh);
    
    // 3 * sizeof(int) => skip the 3 integers for dimension and element count
    // all processes fetch the offset array
    byte_offset = 3 * sizeof(int);
    MPI_File_read_at_all(fh, byte_offset, matrix_offset, M+1, MPI_INT, MPI_STATUS_IGNORE);
   
    if (rank == 0)
        printf("Start Partitioning\n");

    // optimal partition is defined as the number of non-zero elements divided
    // by the number of processes
    int optimal_partition = nz / size;

    // used to count how many rows have been collected by a process
    int partition_size = 0;
    // used to memorise where each process starts colelcting matrix rows
    int partition_cut_idx = 0;
    //
    int part_start,part_stop;

    FFGE_submatrix.N = N;
    FFGE_submatrix.M = M;
    FFGE_submatrix.nz = nz;

    // iteratively, each thread collects a number of rows until it possesses at least 
    // 'optimal_partition' number of non-zero elements
    if (rank==0)
        printf("Partition size: %d\n",optimal_partition);

    for (int i = 0; i < size; ++i)
    {
        partition_size = 0;
        if (rank == i)
        {
            part_start = partition_cut_idx;
        }

        while(true)
        {
            
            // compute the number of elements contained in the nex row.
            partition_size += matrix_offset[partition_cut_idx+1] - matrix_offset[partition_cut_idx];
            partition_cut_idx++;

            // check if a process already contains the minimum amount of elements or all elements have been distributed
            if (optimal_partition <= partition_size)
            {
                if (rank == i)
                {
                    part_stop = partition_cut_idx;
                }

                break;
            }
            else if (((i == size-1) && partition_cut_idx == (M+1)))
            {
                if (rank == i)
                {
                    part_stop = partition_cut_idx-1;
                }

                break;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("End Partitioning\n");


    /*
    Now each process knows which rows it will work with/store
    They still have to read the acctual values from the binary file
    */


    if (rank == 0)
        printf("Initialise submatrix!\n");
    
    int num_rows = part_stop - part_start;
    int num_of_nz_val = matrix_offset[part_stop]-matrix_offset[part_start];
    
    FFGE_submatrix.r_start = part_start;
    FFGE_submatrix.r_stop = part_stop;
    FFGE_submatrix.local_nz = num_of_nz_val;

    FFGE_submatrix.off = (int*)malloc((num_rows+1)*sizeof(int));
    FFGE_submatrix.col = (int*)malloc((num_of_nz_val)*sizeof(int));
    FFGE_submatrix.val = (double*)malloc((num_of_nz_val)*sizeof(double));
    
    // store only the important offset values for each process
    for (int i = part_start; i <= part_stop; i++)
    {
        // make sure the indexing for the offset is translated such that the indexing of the submatrix is satisfied
        FFGE_submatrix.off[i-part_start] = matrix_offset[i]-matrix_offset[part_start]; 
    }

    printf("Rank: %d, Offset: %d - %d, gloabl start-stop: %d - %d\n",rank, FFGE_submatrix.off[0],FFGE_submatrix.off[num_rows],FFGE_submatrix.r_start,FFGE_submatrix.r_stop);
    
    if(rank == 0)
        printf("Start fetch remaining data\n");

    // 3 * sizeof(int) => skip the 3 integers for dimension and element count
    // (M+1) * sizeof(int) => skip the M+1 integers that store the offsets
    // matrix_offset[part_start] * sizeof(int) => start reading in column idx from the first row assigned to the process
    byte_offset = 3 * sizeof(int) + (M+1) * sizeof(int) + matrix_offset[part_start] * sizeof(int);
    MPI_File_read_at_all(fh, byte_offset, FFGE_submatrix.col, num_of_nz_val, MPI_INT, MPI_STATUS_IGNORE);

    //printf("Rank: %d, byte_offset: %ld\n",rank,byte_offset);
    MPI_Barrier(MPI_COMM_WORLD);


    // 3 * sizeof(int) => skip the 3 integers for dimension and element count
    // (M+1) * sizeof(int) => skip the M+1 integers that store the offsets
    // nz * sizeof(int) => skip all column indicies
    // matrix_offset[part_start] * sizeof(double) => start reading at the approriate non-zero value
    byte_offset = 3 * sizeof(int) + (M+1) * sizeof(int) + nz * sizeof(int) + matrix_offset[part_start] * sizeof(double);
    MPI_File_read_at_all(fh, byte_offset, FFGE_submatrix.val, num_of_nz_val, MPI_DOUBLE, MPI_STATUS_IGNORE);
    //printf("Rank: %d, byte_offset: %ld\n",rank,byte_offset);
    

    if (rank==0)
        printf("Finish fetch remaining data\n");
    
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("Complete submatrix initialization!\n");

    
    MPI_File_close(&fh);

    if (rank ==0)
    {
        for (int i = 0; i < 5; i++)
        {
            printf("Rank: %d, idx: %d, val: %.12lf\n", rank, i, FFGE_submatrix.val[i]);
        }
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0)
        printf("Start BCSSTK\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // ProcessData struct contains the index ranges that each process treats
    // This data becomes important for the matrix vector multiplication
    ProcessData pd;

    // ProcessData contains 3 values per process
    pd.ranks = (int*)malloc((size)*sizeof(int));
    pd.r_start = (int*)malloc((size)*sizeof(int));
    pd.r_stop = (int*)malloc((size)*sizeof(int));

    
    // initalise the rank idxs (in highnsight not needed)
    for (int i = 0; i < size; i++)
    {
        pd.ranks[i] = i;
    }
    
    // all processes share their row-range among eachother
    MPI_Allgather(&FFGE_submatrix.r_start,1,MPI_INT,pd.r_start,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Allgather(&FFGE_submatrix.r_stop,1,MPI_INT,pd.r_stop,1,MPI_INT,MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        printf("\n");
        for (int i = 0; i < size; i++)
        {
            printf("Rank: %d, start row: %d, end row: %d\n",pd.ranks[i],pd.r_start[i],pd.r_stop[i]);
        }
    }
    
    cg_solution = cg(M,&FFGE_submatrix,b_06,init_06,epsilon,0,rank,size,&pd);

    free(FFGE_submatrix.off);
    free(FFGE_submatrix.col);
    free(FFGE_submatrix.val);
    free(cg_solution);
    free(b_06);
    free(init_06);
    free(matrix_offset);
    free(pd.r_start);
    free(pd.r_stop);
    free(pd.ranks);

    
    if (rank == 0)
        printf("Finished BCSSTK\n");
    
    
    
    MPI_Finalize();
}



/*
Code to produce custome MPI types for more efficient I/O or communications: 

    MPI_Datatype MPI_ENTRY;
    int blocklens[3] = {1,1,1};
    MPI_Aint offsets[3] = {0,4,8};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE}; 
    MPI_Type_create_struct(3, blocklens, offsets, types, &MPI_ENTRY);
    MPI_Type_commit(&MPI_ENTRY);  

    MPI_Type_free(&MPI_ENTRY);
*/