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
def parallel_jaxConstruction(arr):
    
    startingIndex = rank*4
    endingIndex = startingIndex+4
    arr = arr.at[startingIndex:endingIndex].set(rank+1)
    return arr


@jax.jit
def parallel_sum(x):
    # Pass the 'comm' object to mpi4jax
    packedObject = mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)
    return packedObject
    
def parallel_convert(arr):
    GPUPointerArray = cp.from_dlpack(arr,copy=False)
    
    arr_petsc = PETSc.Vec().createWithDLPack(GPUPointerArray, size=arr.shape[0],comm=comm)
    arr_petsc.assemblyBegin()
    arr_petsc.assemblyEnd()
    print(arr_petsc.getOwnershipRange())
    return arr_petsc.getType()

    
data = jnp.zeros((4*size, 1),dtype=jnp.float64) 
result = parallel_jaxConstruction(data)
result = parallel_sum(result)
result = parallel_convert(result)
