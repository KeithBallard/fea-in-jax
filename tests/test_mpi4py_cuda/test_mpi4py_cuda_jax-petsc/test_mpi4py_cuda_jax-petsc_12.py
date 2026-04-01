from mpi4py import MPI
import mpi4jax
import jax
import jax.numpy as jnp
import petsc4py
import cupy as cp
import os


# 1. Initialize MPI communicator via mpi4py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

petsc4py.init(comm=comm)
from petsc4py import PETSc

jax.config.update('jax_enable_x64', True)


localSize = 100
globalSize = localSize*nprocs


@jax.jit
def parallel_jaxConstruction():
    
    localRankValue = jnp.zeros((localSize,1),dtype=jnp.float64)
    return localRankValue

@jax.jit
def parallel_jaxModification(section):
    
    if rank==0:
        section = section.at[:].set(2)
        section = section.at[2].set(4) #just to check things transfer correctly and it doesn't just fill in 
        
    sendout = mpi4jax.bcast(section,0,comm=comm)
    sendout = sendout+rank

    return sendout

def parallel_buildPETScVec(section):
    
    GPUPointerArray = cp.from_dlpack(section)
    GPUPointerArray = GPUPointerArray.astype(cp.float64) #careful that nothing gets garbage disposed
    
    globalSize = section.shape[0]*nprocs

    local_n = globalSize // nprocs


    local_arr = GPUPointerArray
    #createWithDLPack assumes the input array is already the local portion of the distributed vector. It does zero redistribution.
    arr_petsc = PETSc.Vec().createWithDLPack(
        local_arr,
        size=globalSize,
        comm=comm
    )
    arr_petsc.assemblyBegin()
    arr_petsc.assemblyEnd()
    
    return arr_petsc
    

localVec = parallel_jaxConstruction()
localVec = parallel_jaxModification(localVec)
globalVec = parallel_buildPETScVec(localVec)

#globalVec.view()
print("whole array size",globalVec.getSize())
print("local portion",globalVec.getLocalSize())
print("JAX object size",localVec.shape)
print("JAX array device location",localVec.device)
print("PETSc VEC type",globalVec.getType())