#!/bin/bash
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --time=3:00:00

module purge
module load tools/VTune/2020_update3
module load toolchain/intel/2020b
#icc bcsstk16_test.c ../../src/LAS_cpu_parallel.c ../../src/cg_cpu_parallel.c ../../../matrices/matrix_load.c ../../../matrices/mmio.c -lm -qopenmp -o b16.out

#icc bcsstk13_test.c ../../src/LAS_cpu_parallel.c ../../src/cg_cpu_parallel.c ../../../matrices/matrix_load.c ../../../matrices/mmio.c -lm -qopenmp -o b13.out

export OMP_NUM_THREADS=28
icc bcsstk17_test.c ../../src/LAS_cpu_parallel.c ../../src/cg_cpu_parallel.c ../../../matrices/matrix_load.c ../../../matrices/mmio.c -lm -qopenmp -o b17.out

#aps --collection-mode omp ./b16.out
#aps --collection-mode omp ./b13.out
aps --collection-mode omp ./b17.out
