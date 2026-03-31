from mpi4py import MPI
import mpi4jax

import petsc4py
import cupy as cp
import ctypes as ct
import sys
import traceback
import os
import signal

# 1. Initialize MPI communicator via mpi4py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

petsc4py.init(comm=comm)
from petsc4py import PETSc


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="false" #needed to avoid JAX eating all the memory MPI will later need. Probably there's an intelligent way of doing this, but I'm not sure quite what it is yet, y'dig?


import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)


lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.


arraySize = 1000000

@jax.jit
def parallel_jaxCreation():
    return jnp.ones((arraySize,1),dtype=jnp.float64)
    
@jax.jit
def parallel_jaxOneRankChange(arr):
    
    if rank==0:
        arr = arr.at[:].set(2)
        arr = arr.at[3].set(4) #just to check things transfer correctly and it doesn't just fill in 
        
    sendout = mpi4jax.bcast(arr,0,comm=comm)

    return sendout

def parallelBuildVec(arr):
    
    GPUPointerArray = cp.from_dlpack(arr)
    GPUPointerArray = GPUPointerArray.astype(cp.float64) #careful that nothing gets garbage disposed
    
    arrsize = arr.shape[0]
    nprocs = size

    local_n = arrsize // nprocs
    start = rank * local_n
    end = start + local_n

    local_arr = GPUPointerArray[start:end]
    #createWithDLPack assumes the input array is already the local portion of the distributed vector. It does zero redistribution.
    arr_petsc = PETSc.Vec().createWithDLPack( 
        local_arr,
        size=arrsize,
        comm=comm
    )
    arr_petsc.assemblyBegin()
    arr_petsc.assemblyEnd()

    return arr_petsc

def parallelBuildMat(data,rows,cols):
    
    
    GPUdata = cp.from_dlpack(data,copy=False)
    
    if rank != 0:
        GPUdata = cp.from_dlpack(jnp.zeros((arraySize,1),dtype=jnp.float64),copy=False) #this needs to be here as it adds despite my telling it to insert. There's probably a fix somewhere involving a similar breakdown to vec creation
    
    GPUrows = cp.from_dlpack(rows,copy=False)
    GPUcols = cp.from_dlpack(cols,copy=False)

    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes([arraySize, arraySize])

    mat.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)
    mat.setPreallocationCOO(rows, cols)
    
    mat_ptr = ct.c_void_p(mat.handle)  # the low level pointer of the mat object
    coo_ptr = ct.c_void_p(GPUdata.data.ptr)  # the pointer to GPU memory

   
    MatSetValuesCOO(mat_ptr, coo_ptr, PETSc.InsertMode.INSERT_ALL)

    print(mat.getOwnershipRange())

    mat.assemblyBegin()
    mat.assemblyEnd()

    return mat

def parallelMatVec(mat,vec):

    outputVec = vec.duplicate()
    
    ksp = PETSc.KSP().create()
    ksp.setOperators(mat)
    ksp.setType("cg")
    ksp.setConvergenceHistory()
    ksp.getPC().setType("none")

    ksp.solve(vec,outputVec)
    #outputVec.view()

try:
    vals = parallel_jaxCreation()
    vals = parallel_jaxOneRankChange(vals)

    rowcols = jnp.arange(arraySize,dtype=jnp.int64)

    mat = parallelBuildMat(vals,rowcols,rowcols)

        
    vec = parallel_jaxCreation()
    vec = parallelBuildVec(vec)

    parallelMatVec(mat,vec)
    print("reached")
except Exception as e:  
    if rank==0:
        print("exception",e)
        traceback.print_exc()
    
    comm.Abort(1)
    os.kill(os.getpid(),signal.SIGKILL) #this is essential as otherwise python will hold on to memory it's not supposed to and force the user (you) to call to sudo pkill -f python3.12




comm.Abort(1)
os.kill(os.getpid(),signal.SIGKILL)
