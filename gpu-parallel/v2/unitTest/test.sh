#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu 

module purge
module load compiler/GCC
module load system/CUDA

# test 1
echo test1
nvcc lcv.cu -o main1
mv main1 ./bin
./bin/main1

# test 2
echo test2
nvcc -c ldv.cu -o ldv.o
mv ldv.o ./objects
gcc -c ../../matrices/matrix_load.c -o matrix_load.o
mv matrix_load.o ./objects
gcc -c ../../matrices/mmio.c -o mmio.o
mv mmio.o ./objects

nvcc -o main2 ./objects/ldv.o ./objects/matrix_load.o ./objects/mmio.o
mv main2 ./bin
./bin/main2

# test 3
echo test3
nvcc -c vec_ad.cu -o vec_ad.o
mv vec_ad.o ./objects
nvcc -c ../cuda_practice/vector_add.cu -o vector_add.o
mv vector_add.o ./objects

nvcc -o main3 ./objects/vec_ad.o ./objects/vector_add.o 
mv main3 ./bin
./bin/main3

# test 4
echo test4
nvcc -c cg_m.cu -o cg_m.o
mv cg_m.o ./objects
nvcc -c ../src/LAS_gpu_parallel.cu -o LAS_gpu_parallel.o
mv LAS_gpu_parallel.o ./objects
nvcc -c ../src/cg_gpu_parallel.cu -o cg_gpu_parallel.o
mv cg_gpu_parallel.o ./objects

nvcc -o main4 ./objects/cg_m.o ./objects/cg_gpu_parallel.o ./objects/LAS_gpu_parallel.o
mv main4 ./bin
./bin/main4