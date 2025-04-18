#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu  
module purge
module load compiler/GCC
module load system/CUDA

gcc -c bcsstk06_test.c -o bcsstk06.o
gcc -c ../../matrices/matrix_load.c -o matrix_load.o
gcc -c ../../matrices/mmio.c -o mmio.o

nvcc -c ../src/LAS_gpu_parallel.cu -o LAS_gpu_parallel.o
nvcc -c ../src/cg_gpu_parallel.cu -o cg_gpu_parallel.o

nvcc -o main bcsstk06.o matrix_load.o mmio.o LAS_gpu_parallel.o cg_gpu_parallel.o -lm

