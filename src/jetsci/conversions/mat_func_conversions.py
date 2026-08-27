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
from time import perf_counter

import cupy as cp
import jax
import jax.numpy as jnp
from petsc4py import PETSc

from .vec_func_conversions import petsc_vec_to_jax_array

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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class COOData:
    """JAX-visible COO matrix data."""

    shape: jax.Array
    vals: jax.Array
    rows: jax.Array
    cols: jax.Array

    def tree_flatten(self):
        return (self.shape, self.vals, self.rows, self.cols), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)



def convert_jax_dense_mat_to_coo_data(dense_mat: jnp.ndarray) -> COOData:
    """Convert a dense rank-2 JAX matrix into COOData.

    This testing/convenience path stores every dense entry, including zeros.
    The production sparse path should prefer a JAX function that returns
    `COOData` directly, especially for large systems.
    """
    mat = jnp.asarray(dense_mat)
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


def convert_jax_sparse_coo_to_coo_data(sparse_mat) -> COOData:
    """Convert a JAX sparse COO matrix into the callback COOData convention."""
    return COOData(
        shape=jnp.asarray(sparse_mat.shape, dtype=jnp.int64),
        vals=sparse_mat.data,
        rows=jnp.asarray(sparse_mat.row, dtype=jnp.int32),
        cols=jnp.asarray(sparse_mat.col, dtype=jnp.int32),
    )


def convert_jax_mat_to_coo_data(mat) -> COOData:
    """Accept COOData, JAX sparse COO, or dense rank-2 arrays as COOData."""
    if isinstance(mat, COOData):
        return mat
    if all(hasattr(mat, field) for field in ("shape", "vals", "rows", "cols")):
        return mat
    if all(hasattr(mat, field) for field in ("shape", "data", "row", "col")):
        return convert_jax_sparse_coo_to_coo_data(mat)
    return convert_jax_dense_mat_to_coo_data(jnp.asarray(mat))


def _scalar_float(value):
    value = jnp.asarray(value)
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return float(value)


def _scalar_int(value):
    value = jnp.asarray(value)
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return int(value)


def _record_coo_diagnostics(stats, data: COOData, *, prefix="jacobian", near_zero_tol=1e-14):
    """Record cheap COO sanity diagnostics in the callback stats dictionary."""
    if stats is None:
        return

    vals = jnp.asarray(data.vals)
    rows = jnp.asarray(data.rows)
    cols = jnp.asarray(data.cols)
    abs_vals = jnp.abs(vals)
    finite_mask = jnp.isfinite(vals)
    diag_mask = rows == cols
    diag_abs_vals = jnp.where(diag_mask, abs_vals, jnp.inf)
    offdiag_abs_vals = jnp.where(~diag_mask, abs_vals, jnp.inf)
    diag_count = _scalar_int(jnp.sum(diag_mask))
    shape = tuple(int(x) for x in jnp.asarray(data.shape).tolist())

    stats[f"{prefix}_shape"] = shape
    stats[f"{prefix}_nnz"] = int(vals.size)
    stats[f"{prefix}_finite_count"] = _scalar_int(jnp.sum(finite_mask))
    stats[f"{prefix}_nan_count"] = _scalar_int(jnp.sum(jnp.isnan(vals)))
    stats[f"{prefix}_inf_count"] = _scalar_int(jnp.sum(jnp.isinf(vals)))
    stats[f"{prefix}_abs_min"] = _scalar_float(jnp.min(abs_vals)) if vals.size else None
    stats[f"{prefix}_abs_max"] = _scalar_float(jnp.max(abs_vals)) if vals.size else None
    stats[f"{prefix}_diag_entry_count"] = diag_count
    stats[f"{prefix}_near_zero_entry_count"] = _scalar_int(jnp.sum(abs_vals <= near_zero_tol))
    stats[f"{prefix}_near_zero_diag_count"] = _scalar_int(
        jnp.sum(diag_mask & (abs_vals <= near_zero_tol))
    )
    stats[f"{prefix}_diag_abs_min"] = (
        _scalar_float(jnp.min(diag_abs_vals)) if diag_count else None
    )
    stats[f"{prefix}_diag_abs_max"] = (
        _scalar_float(jnp.max(jnp.where(diag_mask, abs_vals, -jnp.inf)))
        if diag_count
        else None
    )
    stats[f"{prefix}_offdiag_abs_min"] = (
        _scalar_float(jnp.min(offdiag_abs_vals)) if vals.size > diag_count else None
    )


def _mat_set_values_coo(mat, vals):
    """Sets the values for an existing PETSc Mat given values as a CuPy array"""
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


def __assign_petsc_mat_from_coo_data(
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


def _assign_petsc_mat_from_coo_data_prealloc(
    mat,
    data: COOData,
    *,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Assign COOData after setting sizes, type, and COO preallocation."""
    __assign_petsc_mat_from_coo_data(
        mat,
        data,
        set_preallocation=True,
        mat_type=mat_type,
    )


def _assign_petsc_mat_from_coo_data_no_prealloc(mat, data: COOData):
    """Assign COO values into a PETSc Mat whose COO pattern is already set."""
    __assign_petsc_mat_from_coo_data(
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

    @staticmethod
    def _same_device_array(lhs, rhs):
        lhs_cupy = cp.from_dlpack(lhs, copy=False)
        rhs_cupy = cp.from_dlpack(rhs, copy=False)
        return bool(cp.array_equal(lhs_cupy, rhs_cupy).item())

    def needs_preallocation(self, mat, data: COOData):
        handle = int(mat.handle)
        previous_pattern = self.patterns_by_handle.get(handle)
        current_pattern = (data.shape, data.rows, data.cols)
        if previous_pattern is None:
            self.patterns_by_handle[handle] = current_pattern
            return True

        previous_shape, previous_rows, previous_cols = previous_pattern
        same_shape = self._same_device_array(previous_shape, data.shape)
        same_rows = self._same_device_array(previous_rows, data.rows)
        same_cols = self._same_device_array(previous_cols, data.cols)
        same_pattern = same_shape and same_rows and same_cols
        if same_pattern:
            return False

        self.patterns_by_handle[handle] = current_pattern
        return True


def evaluate_jax_dense_jac_to_coo(jax_mat_func, X):
    """Evaluate a dense JAX matrix function on PETSc Vec input as COOData."""
    with _nvtx_range("snes_petsc_vec_to_jax"):
        x = petsc_vec_to_jax_array(X)
    with _nvtx_range("snes_jax_matrix_function"):
        dense_mat = jax_mat_func(x)
    #print(
    #    "jetsci.mat_callback: dense jacobian result shape/dtype",
    #    getattr(dense_mat, "shape", None),
    #    getattr(dense_mat, "dtype", None),
    #)
    #print("jetsci.mat_callback: dense jacobian result", dense_mat)
    with _nvtx_range("snes_dense_mat_to_coo_data"):
        data = convert_jax_dense_mat_to_coo_data(dense_mat)
    return data


def evaluate_jax_coo_jac_to_coo(jax_coo_func, X):
    """Evaluate a JAX COOData-producing function on PETSc Vec input."""
    with _nvtx_range("snes_petsc_vec_to_jax"):
        x = petsc_vec_to_jax_array(X)
    with _nvtx_range("snes_jax_coo_matrix_function"):
        data = jax_coo_func(x)
    #print(
    #    "jetsci.mat_callback: COO jacobian result shape",
    #    getattr(data, "shape", None),
    #)
    #print("jetsci.mat_callback: COO jacobian vals", getattr(data, "vals", None))
    #print("jetsci.mat_callback: COO jacobian rows", getattr(data, "rows", None))
    #print("jetsci.mat_callback: COO jacobian cols", getattr(data, "cols", None))
    return data


def assign_petsc_mat_pair_from_coo(
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
        __assign_petsc_mat_from_coo_data(
            J,
            data,
            set_preallocation=set_j_preallocation,
            mat_type=mat_type,
        )
    if P is not None and P.handle != J.handle:
        with _nvtx_range("snes_assign_preconditioner_mat_direct"):
            __assign_petsc_mat_from_coo_data(
                P,
                data,
                set_preallocation=set_p_preallocation,
                mat_type=mat_type,
            )


def assign_petsc_mat_pair_from_coo_fixed_pattern(
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
        assign_petsc_mat_pair_from_coo(
            J,
            P,
            data,
            set_j_preallocation=set_j_preallocation,
            set_p_preallocation=set_p_preallocation,
            mat_type=mat_type,
        )


def assign_petsc_mat_pair_from_coo_pattern_aware(
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
        assign_petsc_mat_pair_from_coo(
            J,
            P,
            data,
            set_j_preallocation=set_j_preallocation,
            set_p_preallocation=set_p_preallocation,
            mat_type=mat_type,
        )


def convert_jax_dense_mat_func_to_petsc_mat_func(
    jax_mat_func,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Convert a state-only dense JAX matrix function into a PETSc Mat callback.

    PETSc/SNES calls the returned function as `jacobian(snes, X, J, P, args)`.
    `jax_mat_func` is expected to return a rank-2 JAX array. This is convenient
    for testing with `jax.jacfwd(residual_func)`, but it is not the sparse path
    we ultimately want for large systems.
    """

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        data = evaluate_jax_dense_jac_to_coo(jax_mat_func, X)

        assign_petsc_mat_pair_from_coo(
            J,
            P,
            data,
            set_j_preallocation=set_preallocation,
            set_p_preallocation=set_preallocation,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convert_jax_coo_mat_func_to_petsc_mat_func(
    jax_coo_func,
    *,
    set_preallocation=True,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
):
    """Convert a JAX COOData-producing function into a direct PETSc Mat callback."""

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        data = evaluate_jax_coo_jac_to_coo(jax_coo_func, X)

        assign_petsc_mat_pair_from_coo(
            J,
            P,
            data,
            set_j_preallocation=set_preallocation,
            set_p_preallocation=set_preallocation,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convert_jax_dense_mat_func_to_petsc_mat_func_fixed_pattern(
    jax_mat_func,
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
        data = evaluate_jax_dense_jac_to_coo(jax_mat_func, X)
        assign_petsc_mat_pair_from_coo_fixed_pattern(
            J,
            P,
            data,
            state=state,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convert_jax_coo_mat_func_to_petsc_mat_func_fixed_pattern(
    jax_coo_func,
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
        data = evaluate_jax_coo_jac_to_coo(jax_coo_func, X)

        assign_petsc_mat_pair_from_coo_fixed_pattern(
            J,
            P,
            data,
            state=state,
            mat_type=mat_type,
        )
        return None

    return petsc_matrix_function


def convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
    jax_coo_func,
    *,
    mat_type=PETSc.Mat.Type.AIJCUSPARSE,
    state=None,
    stats=None,
):
    """Convert a COOData function into a callback that rebuilds on pattern changes."""
    state = state if state is not None else PatternAwareMatAssignmentState()

    def petsc_matrix_function(snes, X, J, P, petsc_args=None):
        callback_start = perf_counter()
        data = evaluate_jax_coo_jac_to_coo(jax_coo_func, X)
        assign_petsc_mat_pair_from_coo_pattern_aware(
            J,
            P,
            data,
            state=state,
            mat_type=mat_type,
        )
        if stats is not None:
            stats["jacobian_calls"] = stats.get("jacobian_calls", 0) + 1
            stats["jacobian_total_s"] = stats.get("jacobian_total_s", 0.0) + (
                perf_counter() - callback_start
            )
            if stats.get("collect_diagnostics", False):
                _record_coo_diagnostics(stats, data)
        return None

    return petsc_matrix_function
