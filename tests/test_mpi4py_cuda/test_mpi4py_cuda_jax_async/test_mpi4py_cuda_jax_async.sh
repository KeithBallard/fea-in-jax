#!/bin/bash
stderr=$(mpirun -np 4 python test_mpi4py_cuda_jax_async.py 2>&1 >/dev/tty)
[[ $? -eq 0 ]] || echo -e "stderr:\n${stderr}"

