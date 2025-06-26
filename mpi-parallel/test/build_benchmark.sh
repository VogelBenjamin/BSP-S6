#!/bin/bash
#SBATCH -N 2
#SBATCH -c 1
#SBATCH -n 16
#SBATCH --time=22:00:00

module purge
module load toolchain/foss/2020b

mpicc bcsstkxx_test.c -o cg.out -lm  ../src/LAS_mpi.c ../src/cg_mpi.c -o cg.out -lm 
mv cg.out bin

(time srun -n 16 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk06.bin") &> output/bcsstk06_benchmark.txt

(time srun -n 16 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk16.bin") &> output/bcsstk16_benchmark.txt

(time srun -n 16 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk13.bin") &> output/bcsstk13_benchmark.txt

(time srun -n 16 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk17.bin") &> output/bcsstk17_benchmark.txt

#{ time ./pH proton.txt 28 10 0 0 0 3 2 4 7 4 2 ;} 2>> output/timeCompare.txt