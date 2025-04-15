#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu  
module purge
module load compiler/GCC
module load system/CUDA

gcc -c ../../../matrices/matrix_load.c -o matrix_load.o
gcc -c ../../../matrices/mmio.c -o mmio.o

# bccstk06
echo Building bccstk06
gcc -c bcsstk06_test.c -o bcsstk06.o

nvcc -c ../src/LAS_gpu_parallel.cu -o LAS_gpu_parallel.o
nvcc -c ../src/cg_gpu_parallel.cu -o cg_gpu_parallel.o

nvcc -o main_06.out bcsstk06.o matrix_load.o mmio.o LAS_gpu_parallel.o cg_gpu_parallel.o -lm
mv main_06.out ./bin

# bccstk16
echo Building bccstk16
gcc -c bcsstk16_test.c -o bcsstk16.o

nvcc -c ../src/LAS_gpu_parallel.cu -o LAS_gpu_parallel.o
nvcc -c ../src/cg_gpu_parallel.cu -o cg_gpu_parallel.o

nvcc -o main_16.out bcsstk16.o matrix_load.o mmio.o LAS_gpu_parallel.o cg_gpu_parallel.o -lm

mv main_16.out ./bin

#bccstk13
echo Building bccstk13
gcc -c bcsstk13_test.c -o bcsstk13.o

nvcc -c ../src/LAS_gpu_parallel.cu -o LAS_gpu_parallel.o
nvcc -c ../src/cg_gpu_parallel.cu -o cg_gpu_parallel.o 
nvcc -o main_13.out bcsstk13.o matrix_load.o mmio.o LAS_gpu_parallel.o cg_gpu_parallel.o -lm                                 
mv main_13.out ./bin

# bccstk17
echo Building bccstk17
gcc -c bcsstk17_test.c -o bcsstk17.o

nvcc -c ../src/LAS_gpu_parallel.cu -o LAS_gpu_parallel.o
nvcc -c ../src/cg_gpu_parallel.cu -o cg_gpu_parallel.o 
nvcc -o main_17.out bcsstk17.o matrix_load.o mmio.o LAS_gpu_parallel.o cg_gpu_parallel.o -lm                                 
mv main_17.out ./bin

