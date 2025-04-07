#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu  

module load compiler/gcc
module load system/cuda

nvcc -arch=compute_70 bcsstk06_test.cu ../src/LAS_gpu_parallel.cu ../src/cg_gpu_parallel.cu ../../matrices/matrix_load.c ../../matrices/mmio.c -o b06.out -lm
mv b06.out bin

./bin/b06.out