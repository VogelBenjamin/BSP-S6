mpicc bcsstkxx_test.c -o cg.out -lm  ../src/LAS_mpi.c ../src/cg_mpi.c -o cg.out -lm 
mv cg.out bin

(time mpirun -n 3 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk06.bin") &> output/bcsstk06_benchmark.txt

(time mpirun -n 3 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk16.bin") &> output/bcsstk16_benchmark.txt

(time mpirun -n 3 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk13.bin") &> output/bcsstk13_benchmark.txt

(time mpirun -n 3 ./bin/cg.out "../../matrices/matrix_data/binary_bcsstk17.bin") &> output/bcsstk17_benchmark.txt


