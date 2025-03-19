#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time=00:10:00


module load toolchain/foss/2020b

gcc -fopenmp openmp_info_fetch.c

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./a.out

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK * 2))

./a.out
