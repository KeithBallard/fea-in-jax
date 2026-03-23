from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import jax
import jax.numpy as jnp

import cupy as cp
import ctypes as ct


jax.config.update("jax_enable_x64", True)


comm = MPI.COMM_WORLD  #there are two comm object sources. The MPI4py one and the PETSc4py one and unfortunatly they're not the same thing. If we're not careful this can and will cause hanging 
rank = comm.Get_rank()
size = comm.Get_size()

matSize = 10


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
    
    data_gpu = jnp.arange(matSize, dtype=jnp.float64)
    print(f"[Rank {rank}] JAX GPU array: {data_gpu}")

    data_cp = cp.asarray(data_gpu)  # forces real CuPy allocation

    comm.Send([data_cp, MPI.DOUBLE], dest=1, tag=11)
    print(f"[Rank {rank}] sent data to PETSc process")
    print(f"[Rank {rank}] GPU device:", cp.cuda.Device().id)

elif rank == 1:

    data_gpu_1 = cp.empty(matSize, dtype=cp.float64)
    comm.Recv([data_gpu_1, MPI.DOUBLE], source=0, tag=11) #This is where things break if you're not careful about memory assignment. If CUDA falls back to shared memory you'll get an error. See the file environmentSettings for what I'm using to get it working *on my machine*
                                                          #sm,smcuda and extremely brittle right now
    coo_ptr = ct.c_void_p(data_gpu_1.data.ptr)
    
    mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
    mat.setSizes([matSize, matSize])
    
    rows = np.arange(matSize,dtype=jnp.int32)
    cols = np.arange(matSize,dtype=jnp.int32)


    mat.setType(PETSc.Mat.Type.AIJCUSPARSE)
    mat.setPreallocationCOO(rows, cols)
    
    lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
    MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
    MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
    MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.
    mat_ptr = ct.c_void_p(mat.handle)  # the low level pointer of the mat object
    coo_ptr = ct.c_void_p(data_gpu_1.data.ptr)  # the pointer to GPU memory

    

    MatSetValuesCOO(mat_ptr, coo_ptr, PETSc.InsertMode.INSERT_ALL)
    
    print(f"[Rank {rank}] GPU device:", cp.cuda.Device().id)


comm.Barrier()
if rank == 0:
    print("[Rank 0] done cleanly")
elif rank == 1:
    print("[Rank 1] done cleanly")