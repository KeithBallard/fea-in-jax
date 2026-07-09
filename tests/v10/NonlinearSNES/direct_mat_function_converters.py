"""Direct-DLPack matrix-function converters for PETSc SNES callbacks.

This module is the PETSc-as-outer-driver Mat path without buffer callbacks.
PETSc supplies an input Vec and output Mat objects, a JAX function computes a
matrix or COO data, and the result is assigned into PETSc-owned Mat storage
through CuPy/DLPack device views and `MatSetValuesCOO`.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes as ct
from dataclasses import dataclass, field

import cupy as cp
import jax
import jax.numpy as jnp
from petsc4py import PETSc

from ..JaxCallsPETSc.linear_methods import COOData
from .direct_vec_function_converters import petscVecToJAX

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


def convertJaxMatToCOOData(mat):
    """Convert a dense rank-2 JAX matrix into COOData.

    This testing/convenience path stores every dense entry, including zeros.
    The production sparse path should prefer a JAX function that returns
    `COOData` directly, especially for large systems.
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


def assignPETScMatFromCOODataDirect(
    mat,
    data: COOData,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign JAX-visible COOData into a PETSc Mat without buffer_callback."""
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


def assignPETScMatFromCOODataDirectRebuild(
    mat,
    data: COOData,
    *,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData after setting sizes, type, and COO preallocation."""
    assignPETScMatFromCOODataDirect(
        mat,
        data,
        set_preallocation=True,
        mat_type=mat_type,
    )


def assignPETScMatFromCOODataDirectValuesOnly(mat, data: COOData):
    """Assign COO values into a PETSc Mat whose COO pattern is already set."""
    assignPETScMatFromCOODataDirect(
        mat,
        data,
        set_preallocation=False,
        mat_type=None,
    )


@dataclass
class FixedPatternMatAssignmentState:
    """Track which PETSc Mat handles have already had COO preallocation set."""

    preallocated_handles: set[int] = field(default_factory=set)

    def needs_preallocation(self, mat):
        handle = int(mat.handle)
        if handle in self.preallocated_handles:
            return False
        self.preallocated_handles.add(handle)
        return True


@dataclass
class PatternAwareMatAssignmentState:
    """Track COO patterns per PETSc Mat handle and request rebuilds on change."""

    patterns_by_handle: dict[int, tuple[jax.Array, jax.Array, jax.Array]] = field(default_factory=dict)

    def needs_preallocation(self, mat, data: COOData):
        handle = int(mat.handle)
        previous_pattern = self.patterns_by_handle.get(handle)
        current_pattern = (data.shape, data.rows, data.cols)
        if previous_pattern is None:
            self.patterns_by_handle[handle] = current_pattern
            return True

        previous_shape, previous_rows, previous_cols = previous_pattern
        same_pattern = (
            bool(jnp.array_equal(previous_shape, data.shape).block_until_ready())
            and bool(jnp.array_equal(previous_rows, data.rows).block_until_ready())
            and bool(jnp.array_equal(previous_cols, data.cols).block_until_ready())
        )
        if same_pattern:
            return False

        self.patterns_by_handle[handle] = current_pattern
        return True


def evaluateJAXMatFuncAsCOOData(jax_mat_func, X, args=None):
    """Evaluate a dense JAX matrix function on PETSc Vec input as COOData."""
    with _nvtx_range("snes_petsc_vec_to_jax"):
        x = petscVecToJAX(X)
    with _nvtx_range("snes_jax_matrix_function"):
        dense_mat = jax_mat_func(x, args) if args is not None else jax_mat_func(x)
    with _nvtx_range("snes_dense_mat_to_coo_data"):
        return convertJaxMatToCOOData(dense_mat)


def evaluateJAXCOOFuncAsCOOData(jax_coo_func, X, args=None):
    """Evaluate a JAX COOData-producing function on PETSc Vec input."""
    with _nvtx_range("snes_petsc_vec_to_jax"):
        x = petscVecToJAX(X)
    with _nvtx_range("snes_jax_coo_matrix_function"):
        return jax_coo_func(x, args) if args is not None else jax_coo_func(x)


def assignPETScMatPairFromCOODataDirect(
    J,
    P,
    data: COOData,
    *,
    set_j_preallocation=True,
    set_p_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData into SNES Jacobian/preconditioner Mat objects."""
    with _nvtx_range("snes_assign_jacobian_mat_direct"):
        assignPETScMatFromCOODataDirect(
            J,
            data,
            set_preallocation=set_j_preallocation,
            mat_type=mat_type,
        )
    if P is not None and P.handle != J.handle:
        with _nvtx_range("snes_assign_preconditioner_mat_direct"):
            assignPETScMatFromCOODataDirect(
                P,
                data,
                set_preallocation=set_p_preallocation,
                mat_type=mat_type,
            )


def assignPETScMatPairFromCOODataDirectFixedPattern(
    J,
    P,
    data: COOData,
    *,
    state: FixedPatternMatAssignmentState,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData, preallocating once per PETSc Mat handle."""
    set_j_preallocation = state.needs_preallocation(J)
    set_p_preallocation = True
    if P is not None and P.handle != J.handle:
        set_p_preallocation = state.needs_preallocation(P)

    with _nvtx_range("snes_assign_mat_pair_direct_fixed_pattern"):
        assignPETScMatPairFromCOODataDirect(
            J,
            P,
            data,
            set_j_preallocation=set_j_preallocation,
            set_p_preallocation=set_p_preallocation,
            mat_type=mat_type,
        )


def assignPETScMatPairFromCOODataDirectPatternAware(
    J,
    P,
    data: COOData,
    *,
    state: PatternAwareMatAssignmentState,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData, rebuilding only when the COO pattern changes."""
    set_j_preallocation = state.needs_preallocation(J, data)
    set_p_preallocation = True
    if P is not None and P.handle != J.handle:
        set_p_preallocation = state.needs_preallocation(P, data)

    with _nvtx_range("snes_assign_mat_pair_direct_pattern_aware"):
        assignPETScMatPairFromCOODataDirect(
            J,
            P,
            data,
            set_j_preallocation=set_j_preallocation,
            set_p_preallocation=set_p_preallocation,
            mat_type=mat_type,
        )


def convertJAXMatFuncToPETScMatFuncDirect(
    jax_mat_func,
    args=None,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Convert a dense JAX matrix function into a direct PETSc Mat callback.

    PETSc/SNES calls the returned function as `jacobian(snes, X, J, P, args)`.
    `jax_mat_func` is expected to return a rank-2 JAX array. This is convenient
    for testing with `jax.jacfwd(residual_func)`, but it is not the sparse path
    we ultimately want for large systems.
    """

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        active_args = args if args is not None else petsc_args
        data = evaluateJAXMatFuncAsCOOData(jax_mat_func, X, active_args)

        assignPETScMatPairFromCOODataDirect(
            J,
            P,
            data,
            set_j_preallocation=set_preallocation,
            set_p_preallocation=set_preallocation,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convertJAXCOOFuncToPETScMatFuncDirect(
    jax_coo_func,
    args=None,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Convert a JAX COOData-producing function into a direct PETSc Mat callback."""

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        active_args = args if args is not None else petsc_args
        data = evaluateJAXCOOFuncAsCOOData(jax_coo_func, X, active_args)

        assignPETScMatPairFromCOODataDirect(
            J,
            P,
            data,
            set_j_preallocation=set_preallocation,
            set_p_preallocation=set_preallocation,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convertJAXMatFuncToPETScMatFuncDirectFixedPattern(
    jax_mat_func,
    args=None,
    *,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
    state=None,
):
    """Convert a dense JAX matrix function into a fixed-pattern Mat callback.

    The first call for a given PETSc Mat handle sets sizes/type/preallocation.
    Later calls for that handle update values only. This assumes the JAX matrix
    function keeps the same sparsity pattern; it does not verify that.
    """
    state = state if state is not None else FixedPatternMatAssignmentState()

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        active_args = args if args is not None else petsc_args
        data = evaluateJAXMatFuncAsCOOData(jax_mat_func, X, active_args)
        assignPETScMatPairFromCOODataDirectFixedPattern(
            J,
            P,
            data,
            state=state,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convertJAXCOOFuncToPETScMatFuncDirectFixedPattern(
    jax_coo_func,
    args=None,
    *,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
    state=None,
):
    """Convert a COOData-producing JAX function into a fixed-pattern callback.

    The first call for a PETSc Mat handle rebuilds sizes/type/preallocation.
    Later calls for the same handle call only `MatSetValuesCOO` with new values.
    If the sparsity pattern changes, use the rebuild converter or create a new
    state object after explicitly deciding to rebuild.
    """
    state = state if state is not None else FixedPatternMatAssignmentState()

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        active_args = args if args is not None else petsc_args
        data = evaluateJAXCOOFuncAsCOOData(jax_coo_func, X, active_args)

        assignPETScMatPairFromCOODataDirectFixedPattern(
            J,
            P,
            data,
            state=state,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convertJAXCOOFuncToPETScMatFuncDirectPatternAware(
    jax_coo_func,
    args=None,
    *,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
    state=None,
):
    """Convert a COOData function into a callback that rebuilds on pattern changes."""
    state = state if state is not None else PatternAwareMatAssignmentState()

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        active_args = args if args is not None else petsc_args
        data = evaluateJAXCOOFuncAsCOOData(jax_coo_func, X, active_args)
        assignPETScMatPairFromCOODataDirectPatternAware(
            J,
            P,
            data,
            state=state,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


convertJAXMatToCOOData = convertJaxMatToCOOData

__all__ = [
    "FixedPatternMatAssignmentState",
    "PatternAwareMatAssignmentState",
    "assignPETScMatFromCOODataDirect",
    "assignPETScMatFromCOODataDirectRebuild",
    "assignPETScMatFromCOODataDirectValuesOnly",
    "assignPETScMatPairFromCOODataDirect",
    "assignPETScMatPairFromCOODataDirectFixedPattern",
    "assignPETScMatPairFromCOODataDirectPatternAware",
    "convertJAXCOOFuncToPETScMatFuncDirect",
    "convertJAXCOOFuncToPETScMatFuncDirectFixedPattern",
    "convertJAXCOOFuncToPETScMatFuncDirectPatternAware",
    "convertJAXMatFuncToPETScMatFuncDirect",
    "convertJAXMatFuncToPETScMatFuncDirectFixedPattern",
    "convertJAXMatToCOOData",
    "convertJaxMatToCOOData",
    "evaluateJAXCOOFuncAsCOOData",
    "evaluateJAXMatFuncAsCOOData",
]
