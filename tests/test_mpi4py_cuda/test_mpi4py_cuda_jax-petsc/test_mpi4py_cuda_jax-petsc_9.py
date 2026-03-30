from mpi4py import MPI
import mpi4jax
import jax
import jax.numpy as jnp
import petsc4py
import cupy as cp
import ctypes as ct

# 1. Initialize MPI communicator via mpi4py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

petsc4py.init(comm=comm)
from petsc4py import PETSc

jax.config.update('jax_enable_x64', True)


lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.


arraySize = 30000

@jax.jit
def parallel_jaxCreation():
    return jnp.ones((arraySize,1),dtype=jnp.float64)
    
@jax.jit
def parallel_jaxOneRankChange(arr):
    
    if rank==0:
        arr = arr.at[:].set(2)
        
    sendout = mpi4jax.bcast(arr,0,comm=comm)

    return sendout



def parallelBuildMat(data,rows,cols):
    
    
    GPUdata = cp.from_dlpack(data,copy=False)
    
    if rank != 0:
        GPUdata = cp.from_dlpack(jnp.zeros((arraySize,1),dtype=jnp.float64),copy=False) #this needs to be here as it adds despite my telling it to insert
    
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

vec = parallel_jaxCreation()
vec = parallel_jaxOneRankChange(vec)

rowcols = jnp.arange(arraySize,dtype=jnp.int64)

mat = parallelBuildMat(vec,rowcols,rowcols)

    



