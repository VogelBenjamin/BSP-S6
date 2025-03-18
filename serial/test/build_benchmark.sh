#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --time=1:00:00

module purge
module load toolchain/foss/2020b

gcc bcsstk06_test.c ../src/LAS_serial.c ../src/cg_serial.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk06.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk06.out bin
./bin/bcsstk06.out > output/bcsstk06_benchmark.txt

gcc bcsstk16_test.c ../src/LAS_serial.c ../src/cg_serial.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk16.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk16.out bin
./bin/bcsstk16.out > output/bcsstk16_benchmark.txt

gcc bcsstk17_test.c ../src/LAS_serial.c ../src/cg_serial.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk17.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk17.out bin
./bin/bcsstk17.out > output/bcsstk17_benchmark.txt

gcc bcsstk13_test.c ../src/LAS_serial.c ../src/cg_serial.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk13.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk13.out bin
./bin/bcsstk13.out > output/bcsstk13_benchmark.txt

#{ time ./pH proton.txt 28 10 0 0 0 3 2 4 7 4 2 ;} 2>> output/timeCompare.txt