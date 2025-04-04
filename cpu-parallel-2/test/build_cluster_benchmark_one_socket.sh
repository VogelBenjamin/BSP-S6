#!/bin/bash
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --sockets-per-node=1
#SBATCH --time=22:00:00

module purge
module load toolchain/foss/2020b

gcc bcsstk06_test.c ../src/LAS_cpu_parallel.c ../src/cg_cpu_parallel.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk06.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk06.out bin

gcc bcsstk16_test.c ../src/LAS_cpu_parallel.c ../src/cg_cpu_parallel.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk16.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk16.out bin

gcc bcsstk17_test.c ../src/LAS_cpu_parallel.c ../src/cg_cpu_parallel.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk17.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk17.out bin

gcc bcsstk13_test.c ../src/LAS_cpu_parallel.c ../src/cg_cpu_parallel.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o bcsstk13.out -lm -O3 -march=native -fopenmp -ffast-math -funroll-loops 
mv bcsstk13.out bin

#!/bin/bash

# Define the directory to store output files
OUTPUT_DIR="output_one"

# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the list of integers
NUMBERS=(2 28 14 8 4 9 13 27 45)

# Define the list of executables
EXECUTABLES=("bin/bcsstk06.out" "bin/bcsstk13.out" "bin/bcsstk16.out", bin/bcsstk17.out)

# Loop over the integers
for num in "${NUMBERS[@]}"; do
    for exe in "${EXECUTABLES[@]}"; do
        exe_name=$(basename "$exe")
        # Execute the binary file with the integer as an argument
        # Redirect the output to a file named using the executable name and integer
        ( time ./$exe "$num"; ) &> "$OUTPUT_DIR/${exe_name}_$num.txt"
    done
done

echo "Execution complete. Output files are in '$OUTPUT_DIR'"