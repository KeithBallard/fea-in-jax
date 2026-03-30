from mpi4py import MPI
import mpi4jax
import jax
import jax.numpy as jnp
import petsc4py
import cupy as cp

# 1. Initialize MPI communicator via mpi4py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

petsc4py.init(comm=comm)
from petsc4py import PETSc

jax.config.update('jax_enable_x64', True)

# 2. Define a JAX function using mpi4jax
@jax.jit
def parallel_swap(arr):
    arr = arr + rank
    # note: this could also use mpi4jax.sendrecv
    if rank == 0:
        # send, then receive
        mpi4jax.send(arr, dest=1, comm=comm)
        other_arr = mpi4jax.recv(arr, source=1, comm=comm)
        print(rank)
    elif rank == 1:
        # receive, then send
        other_arr = mpi4jax.recv(arr, source=0, comm=comm)
        mpi4jax.send(arr, dest=0, comm=comm)
        print(rank)
    else:
        other_arr = jnp.ones((4,1),dtype=jnp.float64) * -1
        print(rank)

    return other_arr
    
    # Pass the 'comm' object to mpi4jax
    
def parallel_convert(arr):
    GPUPointerArray = cp.from_dlpack(arr,copy=False)
    
    arr_petsc = PETSc.Vec().createWithDLPack(GPUPointerArray, size=arr.shape[0],comm=comm)
    arr_petsc.assemblyBegin()
    arr_petsc.assemblyEnd()
    return arr_petsc.getType()
    

    
data = jnp.ones((4, 1),dtype=jnp.float64) * rank
result = parallel_swap(data)
result = parallel_convert(result)  #okay, this part is working


print(result)