"""Matrix-function converters for PETSc SNES callbacks.

This module is the planned home for converting JAX matrix-output functions
into PETSc Mat-mutating callbacks. It will cover both fixed-sparsity value
updates and pattern-changing rebuild paths.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes as ct

import cupy as cp
import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import buffer_callback
from petsc4py import PETSc

from ..JaxCallsPETSc.linear_methods import COOData
from .vec_function_converters import petscVecToJAX

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


def _mat_set_values_coo(mat, vals):
    lib = ct.CDLL(PETSc.__file__)
    mat_set_values_coo = lib.MatSetValuesCOO
    mat_set_values_coo.restype = ct.c_int
    mat_set_values_coo.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
    err = mat_set_values_coo(
        ct.c_void_p(mat.handle),
        ct.c_void_p(vals.data.ptr),
        PETSc.InsertMode.INSERT_VALUES,
    )
    if err:
        raise RuntimeError(f"MatSetValuesCOO failed with PETSc error code {err}")


def assignPETScMatFromCOOData(
    mat,
    data: COOData,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData values into an existing PETSc Mat.

    This is the matrix analogue of the SNES Vec assignment helper: rows/cols
    define the COO pattern, and values are copied device-to-device from the
    JAX callback buffer into PETSc-owned matrix storage.
    """
    if set_preallocation:
        with _nvtx_range("snes_mat_set_sizes_type_preallocation"):
            mat.setSizes(tuple(data.shape.tolist()))
            if mat_type is not None:
                mat.setType(mat_type)
            mat.setPreallocationCOO(data.rows, data.cols)

    result_info = jax.ShapeDtypeStruct((), jnp.int32)

    def callback(ctx, out, vals):
        del ctx
        with _nvtx_range("snes_mat_values_buffer_callback"):
            vals_cupy = cp.from_dlpack(vals, copy=False)
            with _nvtx_range("snes_mat_set_values_coo"):
                _mat_set_values_coo(mat, vals_cupy)
            cp.asarray(out)[...] = cp.int32(0)

    with _nvtx_range("snes_mat_values_buffer_callback_launch"):
        token = buffer_callback(callback, result_info, vmap_method="sequential")(data.vals)
    token.block_until_ready()


def assignPETScMatFromCOODataDirect(
    mat,
    data: COOData,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData values into a PETSc Mat from ordinary Python control flow.

    This bypasses `buffer_callback`. It is only appropriate when PETSc/Python is
    the outer driver and JAX is being used as a compute engine, because the Mat
    mutation is not represented as a JAX-staged side effect.
    """
    if set_preallocation:
        with _nvtx_range("snes_direct_mat_set_sizes_type_preallocation"):
            mat.setSizes(tuple(data.shape.tolist()))
            if mat_type is not None:
                mat.setType(mat_type)
            mat.setPreallocationCOO(data.rows, data.cols)

    with _nvtx_range("snes_direct_mat_values_ready_and_dlpack"):
        data.vals.block_until_ready()
        vals_cupy = cp.from_dlpack(data.vals, copy=False)

    with _nvtx_range("snes_direct_mat_set_values_coo"):
        _mat_set_values_coo(mat, vals_cupy)


def convertJAXMatFuncToPETScMatFunc(
    jax_mat_func,
    args=None,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
    use_buffer_callback=True,
):
    """Convert a dense JAX matrix function into a PETSc Mat callback.

    PETSc/SNES calls the returned function as `jacobian(snes, X, J, P, args)`.
    `jax_mat_func` is expected to return a rank-2 JAX array. For testing, this
    pairs naturally with `jax.jacfwd(residual_func)`.
    """

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        active_args = args if args is not None else petsc_args
        with _nvtx_range("snes_petsc_vec_to_jax"):
            x = petscVecToJAX(X)
        with _nvtx_range("snes_jax_matrix_function"):
            dense_mat = jax_mat_func(x, active_args) if active_args is not None else jax_mat_func(x)
        with _nvtx_range("snes_dense_mat_to_coo_data"):
            data = convertJaxMatToCOOData(dense_mat)

        assign_mat = assignPETScMatFromCOOData if use_buffer_callback else assignPETScMatFromCOODataDirect

        with _nvtx_range("snes_assign_jacobian_mat"):
            assign_mat(
                J,
                data,
                set_preallocation=set_preallocation,
                mat_type=mat_type,
            )
        if P.handle != J.handle:
            with _nvtx_range("snes_assign_preconditioner_mat"):
                assign_mat(
                    P,
                    data,
                    set_preallocation=set_preallocation,
                    mat_type=mat_type,
                )
        return None

    return petsc_matrix_function


def convertJAXMatValuesFuncToPETScMatFunc(*args, **kwargs):
    """Convert a fixed-pattern JAX matrix-values function into a PETSc Mat callback."""
    raise NotImplementedError("Fixed-pattern JAX matrix values conversion is not implemented yet")


def convertJAXCOOFuncToPETScMatFunc(*args, **kwargs):
    """Convert a JAX COO-producing function into a PETSc Mat callback."""
    raise NotImplementedError("JAX COO function to PETSc Mat conversion is not implemented yet")


# This is just a testing helper. The eventual fixed-pattern path should avoid
# constructing dense COO data when the sparsity pattern is already known.
def convertJaxMatToCOOData(mat):
    """Convert a dense JAX matrix into level-3 COOData.

    This intentionally stores every dense entry, including zeros. That keeps
    the row/column pattern fixed for simple tests using matrices returned by
    `jax.jacfwd`, even though it is not the representation we want for large
    sparse production problems.
    """
    mat = jnp.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"Expected a rank-2 matrix, got shape {mat.shape}")

    n_rows, n_cols = mat.shape
    rows, cols = jnp.meshgrid(
        jnp.arange(n_rows, dtype=jnp.int32),
        jnp.arange(n_cols, dtype=jnp.int32),
        indexing="ij",
    )

    return COOData(
        shape=jnp.asarray((n_rows, n_cols), dtype=jnp.int64),
        vals=mat.reshape(-1),
        rows=rows.reshape(-1),
        cols=cols.reshape(-1),
    )


def bundleMemoryVersions(matFuncRebuild,matFuncSet,sparsityPatternChange):

    if sparsityPatternChange:
        return matFuncRebuild

    else:
        return matFuncSet





convertJAXMatToCOOData = convertJaxMatToCOOData

__all__ = [
    "assignPETScMatFromCOOData",
    "assignPETScMatFromCOODataDirect",
    "convertJAXCOOFuncToPETScMatFunc",
    "convertJAXMatFuncToPETScMatFunc",
    "convertJAXMatValuesFuncToPETScMatFunc",
    "convertJAXMatToCOOData",
    "convertJaxMatToCOOData",
]
