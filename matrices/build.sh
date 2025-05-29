gcc main.c matrix_load.c mmio.c -o load.out -lm
mv load.out bin
./bin/load.out 