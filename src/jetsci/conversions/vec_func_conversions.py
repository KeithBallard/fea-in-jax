"""Direct-DLPack vector-function converters for PETSc SNES callbacks.

This module is the PETSc-as-outer-driver Vec path without buffer callbacks.
PETSc supplies input/output Vec objects, a JAX function computes the residual
or vector-valued operation, and the result is assigned back into PETSc-owned
output storage through CuPy/DLPack device views.
"""

from __future__ import annotations

from contextlib import contextmanager

from time import perf_counter

import inspect

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


def petsc_vec_to_jax_array(vec):
    """Create a JAX array view of a PETSc Vec through DLPack."""
    if hasattr(vec, "toDLPack"):
        import cupy as cp

        vec_cupy = cp.from_dlpack(vec.toDLPack(mode="r"))
        return jax.dlpack.from_dlpack(vec_cupy)
    if not hasattr(vec, "__dlpack__"):
        raise TypeError("PETSc Vec does not expose DLPack; direct input path is unavailable")
    return jax.dlpack.from_dlpack(vec)


def jax_array_to_petsc_vec(values):
    """Create a PETSc Vec that views a JAX array through DLPack."""
    import cupy as cp

    if hasattr(values, "block_until_ready"):
        values.block_until_ready()
    if hasattr(values, "__dlpack_device__"):
        try:
            dlpack_device = values.__dlpack_device__()
        except Exception:
            dlpack_device = None
        if dlpack_device is not None and dlpack_device[0] != 1:
            values_cupy = cp.from_dlpack(values, copy=False)
        else:
            # TODO: remove this host-staging fallback once the traced RHS path is
            # fully device-native again.
            values_cupy = cp.asarray(np.asarray(values))
    elif hasattr(values, "__dlpack__"):
        values_cupy = cp.from_dlpack(values, copy=False)
    else:
        # TODO: remove this host-staging fallback once the traced RHS path is
        # fully device-native again.
        values_cupy = cp.asarray(np.asarray(values))
    return PETSc.Vec().createWithDLPack(values_cupy, size=values_cupy.size)


def assign_petsc_vec_from_jax(vec, values):
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


def convert_jax_vec_func_to_petsc_vec_func(jax_func, *, stats=None):
    """Convert a state-only JAX vector function into a PETSc Vec callback.

    PETSc calls the returned function as `function(snes, X, F, args)`.
    `X` is treated as the input vector, `F` is mutated in-place as the output
    vector, and `jax_func` is called as `jax_func(x)`.
    """

    #notice, here it expects 'x' but inside it does not

    def petsc_function(snes, X, F, petsc_args=None):
        callback_start = perf_counter()

        with _nvtx_range("snes_petsc_vec_to_jax"):
            x = petsc_vec_to_jax_array(X)
        with _nvtx_range("snes_jax_vec_function"):
            values = jax_func(x) #TODO: IMPORTANT okay, something is strange. Why is this saying there's already an argument in it?
        #print("jetsci.vec_callback: residual result shape/dtype", getattr(values, "shape", None), getattr(values, "dtype", None))
        #print("jetsci.vec_callback: residual result", values)
        with _nvtx_range("snes_assign_vec_direct_dlpack"):
            assign_petsc_vec_from_jax(F, values)
        if stats is not None:
            stats["residual_calls"] = stats.get("residual_calls", 0) + 1
            stats["residual_total_s"] = stats.get("residual_total_s", 0.0) + (
                perf_counter() - callback_start
            )
        return None

    return petsc_function
