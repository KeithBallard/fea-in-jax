"""Vector-function converters for PETSc SNES callbacks.

These helpers adapt a pure JAX vector function to PETSc's in-place Vec callback
style. The current path is intentionally GPU/DLPack-only: it does not fall back
to NumPy host copies.
"""

from __future__ import annotations

from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import buffer_callback
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
    """Convert a PETSc Vec to a JAX array through the DLPack protocol."""
    if not hasattr(vec, "__dlpack__"):
        raise TypeError("PETSc Vec does not expose __dlpack__; no-copy input path is unavailable")
    return jax.dlpack.from_dlpack(vec, copy=False)


def jaxArrayToPETScVec(values):
    """Create a PETSc Vec that views a JAX array through DLPack."""
    import cupy as cp

    values.block_until_ready()
    values_cupy = cp.from_dlpack(values, copy=False)
    return PETSc.Vec().createWithDLPack(values_cupy, size=values_cupy.size)


def assignPETScVecFromJAXBuffer(vec, values):
    """Assign JAX values into an existing PETSc Vec through buffer_callback.

    This mirrors the JAX->KSP solve return path: wrap PETSc Vec device memory
    as a CuPy array and copy from the JAX callback buffer into that view.
    """
    result_info = jax.ShapeDtypeStruct((), jnp.int32)

    def callback(ctx, out, values_buffer):
        del ctx
        import cupy as cp

        values_cupy = cp.from_dlpack(values_buffer, copy=False)
        ptr = vec.getCUDAHandle()
        length = vec.getSize()
        nbytes = length * values_cupy.dtype.itemsize
        vec_cupy = cp.ndarray(
            (length,),
            dtype=values_cupy.dtype,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, nbytes, vec), 0),
        )

        vec_cupy[...] = values_cupy.reshape(vec_cupy.shape)
        cp.asarray(out)[...] = cp.int32(0)

    token = buffer_callback(callback, result_info, vmap_method="sequential")(values)
    token.block_until_ready()


def assignPETScVecFromJAXDirect(vec, values):
    """Assign JAX values into an existing PETSc Vec without buffer_callback.

    This is for PETSc-as-outer-driver code. It waits for the JAX result, wraps
    it as a CuPy device array, wraps PETSc Vec memory as a CuPy device array,
    and performs a device-to-device assignment into PETSc-owned storage.
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


def convertJAXVecFuncToPETScVecFunc(
    jax_func,
    args=None,
    *,
    use_buffer_callback=True,
):
    """Convert a pure JAX vector function into a PETSc Vec callback.

    PETSc calls the returned function as `function(snes, X, F, args)`. `X` is
    the PETSc input vector and `F` is PETSc's output vector. The returned
    callback mutates `F`; rebinding `F` to a new Vec will not update SNES.
    """

    def petsc_function(snes, X, F, petsc_args=None):
        active_args = args if args is not None else petsc_args
        with _nvtx_range("snes_petsc_vec_to_jax"):
            x = petscVecToJAX(X)
        with _nvtx_range("snes_jax_vec_function"):
            values = jax_func(x, active_args) if active_args is not None else jax_func(x)

        if use_buffer_callback:
            with _nvtx_range("snes_assign_vec_buffer_callback"):
                assignPETScVecFromJAXBuffer(F, values)
        else:
            with _nvtx_range("snes_assign_vec_direct_dlpack"):
                assignPETScVecFromJAXDirect(F, values)
        return None

    return petsc_function


__all__ = [
    "assignPETScVecFromJAXDirect",
    "assignPETScVecFromJAXBuffer",
    "convertJAXVecFuncToPETScVecFunc",
    "jaxArrayToPETScVec",
    "petscVecToJAX",
]
