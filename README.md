# Parallel code branch
This branch contains a code for MPI+CUDA program and also the pure MPI program.

To build CUDA program on Polus machine, run:

```
make ARCH=sm_60 HOST_COMP=mpicxx
```

And for execution:

```
./run_mpi_cu
```
