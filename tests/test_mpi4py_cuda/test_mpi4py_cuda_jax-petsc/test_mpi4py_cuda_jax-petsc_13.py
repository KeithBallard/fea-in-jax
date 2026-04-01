import petsc4py
import cupy as cp
import os
import ctypes as ct

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="false"

from mpi4py import MPI
import mpi4jax
import jax
import jax.numpy as jnp

# 1. Initialize MPI communicator via mpi4py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

petsc4py.init(comm=comm)
from petsc4py import PETSc

lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.


jax.config.update('jax_enable_x64', True)


localSize = 1000
globalSize = localSize*nprocs


@jax.jit
def parallel_jaxConstruction():
    
    localRankValue = jnp.zeros((localSize,1),dtype=jnp.float64)
    return localRankValue

@jax.jit
def parallel_jaxBuildVals(section):
    
    if rank==0:
        section = section.at[:].set(2)
        section = section.at[2].set(4) #just to check things transfer correctly and it doesn't just fill in 
        
    sendout = mpi4jax.bcast(section,0,comm=comm)
    sendout = sendout+rank

    return sendout

@jax.jit
def paralell_buildIndex():
    start = rank*localSize
    end = (start+localSize)
    return jnp.arange(start,end,dtype=jnp.float32)



def parallel_fillPETScMat(localVals,localRows,localCols):
    
    GPUdata = cp.from_dlpack(localVals,copy=False)
    
    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes([globalSize, globalSize])

    mat.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)

    print(localRows.device)
    print(localCols.device)

    mat.setPreallocationCOO(localRows, localCols)
    
    mat_ptr = ct.c_void_p(mat.handle)  # the low level pointer of the mat object
    coo_ptr = ct.c_void_p(GPUdata.data.ptr)  # the pointer to GPU memory

   
    MatSetValuesCOO(mat_ptr, coo_ptr, PETSc.InsertMode.INSERT_ALL)
    
    mat.assemblyBegin()
    mat.assemblyEnd()
    
    return mat
    
    


localVec = parallel_jaxConstruction()
localVal = parallel_jaxBuildVals(localVec)
localRows = paralell_buildIndex()
localCols = paralell_buildIndex()

matMPI = parallel_fillPETScMat(localVal,localRows,localRows)

print(matMPI.getSize())