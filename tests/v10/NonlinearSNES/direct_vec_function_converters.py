"""Direct-DLPack vector-function converters for PETSc SNES callbacks.

This module is the PETSc-as-outer-driver Vec path without buffer callbacks.
PETSc supplies input/output Vec objects, a JAX function computes the residual
or vector-valued operation, and the result is assigned back into PETSc-owned
output storage through CuPy/DLPack device views.
"""

from __future__ import annotations

from contextlib import contextmanager

import jax
from petsc4py import PETSc

try:
    from cupyx.profiler import time_range as _cupy_time_range
except Exception:
    _cupy_time_range = None


@contextmanager
def _nvtx_range(name):
    if _cupy_time_range is None:
        with jax.profiler.TraceAnnotation(name):
            yield
    else:
        with _cupy_time_range(name):
            yield


def petscVecToJAX(vec):
    """Create a JAX array view of a PETSc Vec through DLPack."""
    if hasattr(vec, "toDLPack"):
        import cupy as cp

        vec_cupy = cp.from_dlpack(vec.toDLPack(mode="r"),copy=False)
        return jax.dlpack.from_dlpack(vec_cupy,copy=False)
    if not hasattr(vec, "__dlpack__"):
        raise TypeError("PETSc Vec does not expose DLPack; direct input path is unavailable")
    return jax.dlpack.from_dlpack(vec,copy=False)


def jaxArrayToPETScVec(values):
    """Create a PETSc Vec that views a JAX array through DLPack."""
    import cupy as cp

    values.block_until_ready()
    values_cupy = cp.from_dlpack(values, copy=False)
    petscVecObject =  PETSc.Vec().createWithDLPack(values_cupy, size=values_cupy.size)
    print(petscVecObject.getType())
    return petscVecObject


def assignPETScVecFromJAXDirect(vec, values):
    """Assign a JAX vector result into an existing PETSc Vec on device.

    This performs a device-to-device assignment from JAX-owned result storage
    into PETSc-owned output Vec storage. It does not use `buffer_callback` and
    does not intentionally stage through host memory.
    """
    import cupy as cp

    with _nvtx_range("snes_direct_vec_values_ready_and_dlpack"):
        values.block_until_ready()
        values_cupy = cp.from_dlpack(values, copy=False)

    with _nvtx_range("snes_direct_vec_assign_to_petsc"):
        ptr = vec.getCUDAHandle()
        length = vec.getSize()
        nbytes = length * values_cupy.dtype.itemsize
        vec_cupy = cp.ndarray(
            (length,),
            dtype=values_cupy.dtype,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, nbytes, vec), 0),
        )
        vec_cupy[...] = values_cupy.reshape(vec_cupy.shape)


def convertJAXVecFuncToPETScVecFuncDirect(jax_func, args=None):
    """Convert a JAX vector function into a direct-DLPack PETSc Vec callback.

    PETSc calls the returned function as `function(snes, X, F, args)`.
    `X` is treated as the input vector, `F` is mutated in-place as the output
    vector, and `jax_func` is called as either `jax_func(x)` or
    `jax_func(x, args)`.
    """

    def petsc_function(snes, X, F, petsc_args=None):
        active_args = args if args is not None else petsc_args
        with _nvtx_range("snes_petsc_vec_to_jax"):
            x = petscVecToJAX(X)
        with _nvtx_range("snes_jax_vec_function"):
            values = jax_func(x, active_args) if active_args is not None else jax_func(x)
        with _nvtx_range("snes_assign_vec_direct_dlpack"):
            assignPETScVecFromJAXDirect(F, values)
        return None

    return petsc_function


__all__ = [
    "assignPETScVecFromJAXDirect",
    "convertJAXVecFuncToPETScVecFuncDirect",
    "jaxArrayToPETScVec",
    "petscVecToJAX",
]
