# BSP-S6
### By Benjamin Vogel (University of Luxembourg 24/25)

This repository presents my work on analysing the impact of parallelization techniques and data structures on the conjguate gradient method for solving large scale linear systems.

### Requirements
Compiler version:
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

mpiexec version:
HYDRA build details:
    Version:                                 4.0
    Release Date:                            Fri Jan 21 10:42:29 CST 2022

cuda version:


### where to locate batch scripts to execute the tests yourself on cluster

#### serial
```\js
cd serial/test/
./build_benchmakr.sh
```

#### multithreading
```\js
cd cpu-parallel-2/test/
./build_cluster_benchmark.sh
```

#### cuda
```\js
cd gpu-parallel/v2/test/
./build_test.sh
time ./bin/main_06.out
time ./bin/main_13.out
time ./bin/main_16.out
time ./bin/main_17.out
```

#### MPI
```\js
cd mpi-parallel/test/
./build_test.sh
```
