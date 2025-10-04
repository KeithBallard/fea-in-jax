#!/bin/bash
#mpirun -np 2 python test_mpi4py_cuda_jax.py
stderr=$(mpirun -np 2 python test_mpi4py_cuda_jax.py 2>&1 >/dev/tty)
[[ $? -eq 0 ]] || echo -e "stderr:\n${stderr}"
