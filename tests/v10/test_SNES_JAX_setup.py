"""Prototype converting JAX residual functions into PETSc SNES callbacks.

Run from the PETScJVP root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_SNES_JAX_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import buffer_callback
from petsc4py import PETSc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


jax.config.update("jax_enable_x64", True)


def example_residual(x, args):
    return jnp.array(
        [
            x[0] ** 2 - args[0],
            x[1] ** 2 - args[1],
            x[2] ** 3 - args[2],
        ],
        dtype=x.dtype,
    )


def _petsc_vec_to_jax(vec):
    """Convert a PETSc Vec to a JAX array through DLPack."""
    if not hasattr(vec, "toDLPack"):
        raise TypeError("PETSc Vec does not expose toDLPack; no-copy input path is unavailable")
    return jax.dlpack.from_dlpack(vec.toDLPack())


def _jax_array_to_petsc_vec(values):
    """Create a PETSc Vec that views a JAX array through DLPack."""
    import cupy as cp

    values.block_until_ready()
    values_cupy = cp.from_dlpack(values, copy=False)
    return PETSc.Vec().createWithDLPack(values_cupy, size=values_cupy.size)


def _assign_petsc_vec_from_jax_buffer_callback(vec, values):
    """Assign JAX values into an existing PETSc Vec through buffer_callback.

    This mirrors the JAX->KSP solve return path: wrap PETSc Vec device memory
    as a CuPy array and copy between that view and the JAX callback buffer.
    """
    result_info = jax.ShapeDtypeStruct((), jnp.int32)               #there's got to be a way of doing this without copying

    def callback(ctx, out, values_buffer):
        del ctx
        import cupy as cp #can we move this out?

        values_cupy = cp.from_dlpack(values_buffer, copy=False)
        ptr = vec.getCUDAHandle()
        length = vec.getSize()
        nbytes = length * values_cupy.dtype.itemsize
        vec_cupy = cp.ndarray(
            (length,),
            dtype=values_cupy.dtype,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, nbytes, vec), 0),
        )


        print(vec_cupy.shape)
        print(values_cupy.shape)

        vec_cupy[...] = values_cupy.reshape(vec_cupy.shape) #this performs a copy, can we do without it?

        cp.asarray(out)[...] = cp.int32(0)

    token = buffer_callback(callback, result_info, vmap_method="sequential")(values)
    token.block_until_ready()



#To be clear, this is for vector input/output functions
def convertJaxFuncToPetscFunc(
    jax_func,
    args,
    *,
    use_buffer_callback=True,
):
    """Convert a pure JAX residual `f(x, args)` into a PETSc SNES function.

    PETSc calls the returned function as `function(snes, X, F, args)`. `X` is the
    PETSc input vector and `F` is PETSc's output vector. The returned callback
    must mutate `F`; rebinding `F` to a new Vec will not update SNES.
    """

    def petsc_function(snes, X, F, args):
        del snes
        x = _petsc_vec_to_jax(X)
        residual = jax_func(x, args)

        if not use_buffer_callback:
            raise ValueError("Only the no-copy buffer_callback path is enabled in this test")

        _assign_petsc_vec_from_jax_buffer_callback(F, residual)
        return None

    return petsc_function


def main():
    args = jnp.array([4.0, 1.0, 27.0], dtype=jnp.float64)
    x0 = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float64)
    expected = example_residual(x0, args)

    X = _jax_array_to_petsc_vec(x0)
    F = X.duplicate()
    expected_vec = _jax_array_to_petsc_vec(expected)
    try:
        print("Testing buffer_callback/DLPack JAX residual -> PETSc F transform.")
        callback_func = convertJAXfuncToPETScFunc(
            example_residual,
            args,
            use_buffer_callback=True,
        )
        callback_func(None, X, F)

        diff = F.duplicate()
        F.copy(diff)
        diff.axpy(-1.0, expected_vec)
        error_norm = diff.norm()
        print("PETSc-space error norm:", error_norm)
        assert error_norm < 1e-12
        diff.destroy()
    finally:
        X.destroy()
        F.destroy()
        expected_vec.destroy()


if __name__ == "__main__":
    main()
