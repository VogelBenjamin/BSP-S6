#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu  
module purge
module load compiler/GCC
module load system/CUDA

nvcc -arch=compute_70 bcsstk06_test.cu ../src/LAS_gpu_parallel.cu ../src/cg_gpu_parallel.cu ../../matrices/matrix_load_gpu.cu ../../matrices/mmio_gpu.cu -o b06.out -lm
mv b06.out bin

./bin/b06.out
