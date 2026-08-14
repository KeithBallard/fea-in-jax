
from . import buildKSP_Keith as buildKSP
import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import buffer_callback
import ctypes as ct
import cupy as cp
import time

import petsc4py
from jax._src.lib import xla_client
from petsc4py import PETSc


#Called function that prints the contents of a KSP
def __petsc_print_impl(ctx,out,handle,b):
    ksp = buildKSP.__retrieve_KSP(cp.asarray(handle))
    ksp.view()
    cp.asarray(out)[...] = b
    
#JAX function to print the contents of a KSP given a dictionary ID
def printPETSC(ctx: buildKSP.__CupyCtx,b):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    x= buffer_callback(__petsc_print_impl, result_info, vmap_method="sequential")(ctx.handle,b)  #unsurprisingly this is where things clash between MPI and PETSc
    return x

#Called function to solve Ax=b with a given KSP and what have yous
def __petsc_solve_impl(ctx, out, handle: jnp.ndarray, pcHandle: jnp.ndarray, b: jnp.ndarray):

    GPUPointerArray = cp.from_dlpack(b,copy=False)
    
    print(b.__dlpack_device__())
    print(GPUPointerArray.__dlpack_device__())
    
    b_petsc_1 = PETSc.Vec().createWithDLPack(GPUPointerArray, size=b.shape[0])


    ksp = buildKSP.__retrieve_KSP(cp.asarray(handle))
    pc = buildKSP.__retrieve_PC(cp.asarray(pcHandle))
    ksp.setPC(pc)
    
    x_petsc = PETSc.Vec().create(PETSc.COMM_SELF)
    x_petsc.setType('cuda')         # true GPU vector
    x_petsc.setSizes(b.shape[0])
    x_petsc.setUp()
    x_petsc.set(1.0)

    n = 3
    
    start = time.time()
    
    ksp.solve(b_petsc_1,x_petsc)
    
    print("inner solve time:",time.time()-start)
    
    
    print("exit code",ksp.getConvergedReason())
    print("iterations",ksp.getIterationNumber())
    print("res norm",ksp.getResidualNorm())

    cudahandle = x_petsc.getCUDAHandle()
    ptr = cudahandle         # raw CUDA pointer from PETSc
    length = x_petsc.getSize()
     
    x_gpu = cp.ndarray((length,), dtype=cp.float64 , memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, length*8, x_petsc), 0))

    print("x First 10 elements inside buffer_callback:", x_gpu[0:10])

    x_petsc.destroy()
    b_petsc_1.destroy()
    

    cp.asarray(out)[...] = x_gpu

#A similar function as above but performs the transpose instead
def __petsc_solve_impl_transpose(ctx, out, handle: jnp.ndarray, pcHandle: jnp.ndarray, b: jnp.ndarray):

    GPUPointerArray = cp.from_dlpack(b,copy=False)
    
    print(b.__dlpack_device__())
    print(GPUPointerArray.__dlpack_device__())
    
    b_petsc_1 = PETSc.Vec().createWithDLPack(GPUPointerArray, size=b.shape[0])

    ksp = buildKSP.__retrieve_KSP(cp.asarray(handle))
    pc = buildKSP.__retrieve_PC(cp.asarray(pcHandle))
    ksp.setPC(pc)
    
    x_petsc = PETSc.Vec().create(PETSc.COMM_SELF)
    x_petsc.setType('cuda')         # true GPU vector
    x_petsc.setSizes(b.shape[0])
    x_petsc.setUp()
    x_petsc.set(1.0)
    
    start = time.time()
    
    ksp.solveTranspose(b_petsc_1,x_petsc)
    
    print("inner solve time:",time.time()-start)
    
    
    print("exit code",ksp.getConvergedReason())
    print("iterations",ksp.getIterationNumber())
    print("res norm",ksp.getResidualNorm())

    cudahandle = x_petsc.getCUDAHandle()
    ptr = cudahandle         # raw CUDA pointer from PETSc
    length = x_petsc.getSize()
     
    x_gpu = cp.ndarray((length,), dtype=cp.float64 , memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, length*8, x_petsc), 0))

    print("x First 10 elements inside buffer_callback:", x_gpu[0:10])

    x_petsc.destroy()
    b_petsc_1.destroy()
    

    cp.asarray(out)[...] = x_gpu


#______________________________________________________________________________________________


#Calling function that solves Ax=b
@jax.jit
def __petsc_solve(ctx: buildKSP.__CupyCtx, pcCtx: buildKSP.__CupyCtx, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    x= buffer_callback(__petsc_solve_impl, result_info, vmap_method="sequential")(ctx.handle, pcCtx.handle, b)  #unsurprisingly this is where things clash between MPI and PETSc
    return x

#Calling function that calls the transpose problem
@jax.jit
def __petsc_solve_transpose(ctx: buildKSP.__CupyCtx, pcCtx: buildKSP.__CupyCtx, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    x= buffer_callback(__petsc_solve_impl_transpose, result_info, vmap_method="sequential")(ctx.handle, pcCtx.handle, b)  #unsurprisingly this is where things clash between MPI and PETSc
    return x

#No idea what this is
def _raw_solve_ksp_callback(ID, PC, A, b):
    """Private raw callback solve.

    Level-3 code should call `KSP_solve`, which binds the primitive-backed
    operation and keeps the JAX-visible matrix data available for rules.
    """
    del A
    return __petsc_solve(ID, PC, b)
