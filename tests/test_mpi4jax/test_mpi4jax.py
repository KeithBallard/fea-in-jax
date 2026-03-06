import numpy as np
import jax.numpy as jnp

from mpi4py import MPI
import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# create a new communicator for mpi4jax
comm_jax = comm.Clone()

arr_np = np.random.rand(10, 10)
arr_jax = jnp.zeros((10, 10))

if rank == 0:
    mpi4jax.send(arr_jax, comm=comm_jax)
    comm.send(arr_np)
else:
    arr_jax = mpi4jax.recv(arr_jax, comm=comm_jax)
    arr = comm.recv(arr_np)

comm_jax.Free()
