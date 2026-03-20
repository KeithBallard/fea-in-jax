import jax
from jax import numpy as jnp
import numpy as np

import cupy as cp

from petsc4py import PETSc

jax.config.update("jax_enable_x64", True)


size = 10000000

#put an MPI guard here
GPUarray = jnp.ones((size,1),jnp.float64)

GPUarray = GPUarray*2

GPUPointerArray = cp.from_dlpack(GPUarray,copy=False) #this doesn't seem to be creating a copy nor does it seem to be creating a version on the CPU yet

#CPUarray = np.asarray(GPUPointerArray.get(),dtype=np.float64) This would move onto CPU, but we don't want that yet


rows = jnp.arange(size,dtype=jnp.int32)
cols = jnp.arange(size,dtype=jnp.int32)

#to here. these are only done by rank 0

#rank 0 and 1 do this
comm = PETSc.COMM_SELF
mat = PETSc.Mat().create(comm=comm)
mat.setSizes([size, size])

mat.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)
mat.setPreallocationCOO(rows, cols)
#to here

#rank 0 from here
import ctypes as ct
lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.
mat_ptr = ct.c_void_p(mat.handle)  # the low level pointer of the mat object
coo_ptr = ct.c_void_p(GPUPointerArray.data.ptr)  # the pointer to GPU memory


MatSetValuesCOO(mat_ptr, coo_ptr, PETSc.InsertMode.INSERT_ALL)
#to here

#do assembly here


#verify using an mpi vector