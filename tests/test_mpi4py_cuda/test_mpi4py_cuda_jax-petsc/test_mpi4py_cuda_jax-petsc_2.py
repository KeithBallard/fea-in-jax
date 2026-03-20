from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import jax
import jax.numpy as jnp

import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise RuntimeError("This example requires exactly 2 MPI ranks")

if rank == 0:
    # -----------------------------
    # Process 0: JAX GPU work
    # -----------------------------
    print(f"[Rank {rank}] starting JAX GPU work")
    jax.config.update("jax_enable_x64", True)
    if not jax.devices("gpu"):
        raise RuntimeError("No GPU available for JAX")
    
    data_gpu = jnp.arange(10, dtype=jnp.float64)
    print(f"[Rank {rank}] JAX GPU array: {data_gpu}")

    comm.send(cp.from_dlpack(data_gpu,copy=False), dest=1, tag=11)
    print(f"[Rank {rank}] sent data to PETSc process")

elif rank == 1:

    print(f"[Rank {rank}] waiting for data from JAX process")
    data_gpu = comm.recv(source=0, tag=11)
    print(f"[Rank {rank}] received data: {type(data_gpu)}")

    vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)  # single-rank CPU vector
    vec.setSizes(len(data_gpu))
    vec.setFromOptions()


    start, end = vec.getOwnershipRange()
    vec[start:end] = data_gpu.get()
    vec.assemble()
    print(f"[Rank {rank}] PETSc vector assembled:")
    vec.view()

    vec *= 2.0
    vec.assemble()
    print(f"[Rank {rank}] PETSc vector after scaling by 2:")
    vec.view()


    result_cpu = vec.getArray()
    comm.send(result_cpu, dest=0, tag=22)
    print(f"[Rank {rank}] sent processed data back to JAX process")

if rank == 0:
    result_cpu = comm.recv(source=1, tag=22)
    result_gpu = jnp.array(result_cpu)  # copy back to GPU
    print(f"[Rank {rank}] received processed data on GPU: {result_gpu}")

comm.Barrier()
if rank == 0:
    print("[Rank 0] done cleanly")
elif rank == 1:
    print("[Rank 1] done cleanly")