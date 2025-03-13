gcc LAS_test.c ../src/LAS_serial.c ../src/cg_serial.c -o LAS.out -lm
mv LAS.out bin
./bin/LAS.out


gcc cg_test.c ../src/LAS_serial.c ../src/cg_serial.c ../../matrices/matrix_load.c ../../matrices/mmio.c -o cg.out -lm
mv cg.out bin
./bin/cg.out