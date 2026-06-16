
import buildKSP
import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import buffer_callback
import ctypes as ct
import cupy as cp
import time

import petsc4py
from jax._src.lib import xla_client
from petsc4py import PETSc

def __petsc_solve_impl(ctx, out, handle: jnp.ndarray, b: jnp.ndarray):

    GPUPointerArray = cp.from_dlpack(b,copy=False)
    
    print(b.__dlpack_device__())
    print(GPUPointerArray.__dlpack_device__())
    
    b_petsc_1 = PETSc.Vec().createWithDLPack(GPUPointerArray, size=b.shape[0])
    



    ksp = buildKSP.__retrieve_object(cp.asarray(handle))
    
    x_petsc = PETSc.Vec().create(PETSc.COMM_SELF)
    x_petsc.setType('cuda')         # true GPU vector
    x_petsc.setSizes(b.shape[0])
    x_petsc.setUp()
    x_petsc.set(1.0)

    n = 3
    
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

                      #rtol, atol, dtol, max_it
    ksp.setTolerances(1e-14,1e-10, 100, 100000) #careful with these, petsc is quite a bit more rigorusly demanding than scipy 

    
    ksp.setConvergenceHistory(n)
    
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

def __petsc_solve_impl_transpose(ctx, out, handle: jnp.ndarray, b: jnp.ndarray):

    GPUPointerArray = cp.from_dlpack(b,copy=False)
    
    print(b.__dlpack_device__())
    print(GPUPointerArray.__dlpack_device__())
    
    b_petsc_1 = PETSc.Vec().createWithDLPack(GPUPointerArray, size=b.shape[0])
    



    ksp = buildKSP.__retrieve_object(cp.asarray(handle))
    
    x_petsc = PETSc.Vec().create(PETSc.COMM_SELF)
    x_petsc.setType('cuda')         # true GPU vector
    x_petsc.setSizes(b.shape[0])
    x_petsc.setUp()
    x_petsc.set(1.0)

    n = 3
    
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

                      #rtol, atol, dtol, max_it
    ksp.setTolerances(1e-14,1e-10, 100, 100000) #careful with these, petsc is quite a bit more rigorusly demanding than scipy 

    
    ksp.setConvergenceHistory(n)
    
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

@jax.jit
def __petsc_solve(ctx: buildKSP.__CupyCtx, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    x= buffer_callback(__petsc_solve_impl, result_info, vmap_method="sequential")(ctx.handle, b)  #unsurprisingly this is where things clash between MPI and PETSc
    return x

@jax.jit
def __petsc_solve_transpose(ctx: buildKSP.__CupyCtx, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    x= buffer_callback(__petsc_solve_impl_transpose, result_info, vmap_method="sequential")(ctx.handle, b)  #unsurprisingly this is where things clash between MPI and PETSc
    return x

# Differentiation rules live in primitiveKSP.solver_differentiation.
