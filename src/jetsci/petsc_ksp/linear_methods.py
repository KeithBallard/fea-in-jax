"""Level-3 PETSc methods.

This module is the public-ish method layer for manual PETSc lifecycle control.
It calls the private level-4 primitive/callback implementation and gives the
rest of the project a small, stable vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .options import PETScMethodOptions
from ._primitives import buildKSP_Keith as _petsc_lifetime
from ._primitives import differentiateKSP_Keith as _solve_primitive
from ._primitives import runKSP_Keith as _raw_callbacks


@dataclass(frozen=True)
class COOData:
    """JAX-visible COO data passed into level-3 methods."""

    shape: jax.Array
    vals: jax.Array
    rows: jax.Array
    cols: jax.Array


def to_COOData_object(jax_coo) -> COOData:
    """Convert a JAX sparse COO object to the level-3 tuple convention."""
    if isinstance(jax_coo, COOData):
        return jax_coo
    return COOData(
        shape=jnp.asarray(jax_coo.shape, dtype=jnp.int64),
        vals=jax_coo.data,
        rows=jnp.asarray(jax_coo.row, dtype=jnp.int32),
        cols=jnp.asarray(jax_coo.col, dtype=jnp.int32),
    )


def evaluate_matrix_function(matrix_function, x) -> COOData:
    """Evaluate a matrix-producing callable at state `x`.

    The callable may return a JAX sparse COO matrix or a `COOData` object.
    This is the level-3 entry point we want nonlinear code to use for
    Jacobian-like callables without naming the method after Jacobians.
    """
    return to_COOData_object(matrix_function(x))


def init_matrix_from_coo(jax_coo, options: PETScMethodOptions = PETScMethodOptions()):
    """Create a PETSc Mat from a JAX sparse COO matrix."""
    data = to_COOData_object(jax_coo)
    return init_matrix_from_COOData(data, options)


def init_matrix_from_COOData(data: COOData, options: PETScMethodOptions = PETScMethodOptions()):
    """Create a PETSc Mat from explicit JAX-visible COO arrays."""
    return _petsc_lifetime.linearMatrixInitFromCOOData(
        data.shape,
        data.vals,
        data.rows,
        data.cols,
        constructionOptions=options.matrix_construction_options(),
    )


def init_matrix_from_function(matrix_function, x, options: PETScMethodOptions = PETScMethodOptions()):
    """Create a PETSc Mat by evaluating a matrix-producing callable at `x`.

    In a nonlinear solve this callable will usually be `J_w_dirichlet`, but
    level 3 only requires that it produce a matrix, not that it be a Jacobian.
    """
    data = evaluate_matrix_function(matrix_function, x)
    return init_matrix_from_COOData(data, options)


def update_matrix_values(matrix, vals):
    """Update values in an existing PETSc Mat with the same sparsity pattern."""
    return _petsc_lifetime.linearMatrixUpdate(matrix, vals)


def cleanup_matrix(matrix):
    """Destroy a PETSc Mat created by this layer."""
    return _petsc_lifetime.linearMatrixCleanup(matrix)


def init_pc(matrix, options: PETScMethodOptions = PETScMethodOptions()):
    """Create a PETSc PC for an existing matrix."""
    return _petsc_lifetime.linearPCInit(
        matrix,
        constructionOptions=options.pc_construction_options(),
    )


def cleanup_pc(pc):
    """Destroy a PETSc PC created by this layer."""
    return _petsc_lifetime.linearPCCleanup(pc)


def init_ksp(matrix, options: PETScMethodOptions = PETScMethodOptions()):
    """Create a PETSc KSP for an existing matrix."""
    return _petsc_lifetime.linearKSPInit(
        matrix,
        constructionOptions=options.ksp_construction_options(),
    )


def cleanup_ksp(ksp):
    """Destroy a PETSc KSP created by this layer."""
    return _petsc_lifetime.linearSolverCleanup(ksp)


def solve_ksp(ksp, pc, jax_coo, b):
    """Solve `A x = b` with a live KSP/PC.

    `jax_coo` is intentionally passed even though PETSc already owns a Mat.
    Its values and sparsity are the JAX-visible representation used by the
    private primitive for JVP/VJP rules.
    """
    data = to_COOData_object(jax_coo)
    return solve_ksp_from_coo_data(ksp, pc, data, b)


def solve_ksp_from_coo_data(ksp, pc, data: COOData, b):
    """Solve using explicit JAX-visible COO data."""
    return _solve_primitive._bind_KSP_solve_primitive(
        ksp,
        pc,
        data.shape,
        data.rows,
        data.cols,
        data.vals,
        b,
    )


def solve_ksp_transpose(ksp, pc, b):
    """Direct PETSc transpose solve.

    Most user code should not need this directly. It is exposed at level 3
    because some nonlinear/adjoint experiments may need manual control.
    Automatic VJP rules call the private primitive path instead.
    """
    return _raw_callbacks.__petsc_solve_transpose(ksp, pc, b)


KSP_solve = solve_ksp
KSP_solve_from_coo_data = solve_ksp_from_coo_data
KSP_solve_from_COOData = solve_ksp_from_coo_data
KSP_solve_transpose = solve_ksp_transpose


def solve_once(jax_coo, b, options: PETScMethodOptions = PETScMethodOptions()):
    """Convenience level-3 method: build Mat/PC/KSP, solve once, cleanup."""
    matrix = init_matrix_from_coo(jax_coo, options)
    pc = init_pc(matrix, options)
    ksp = init_ksp(matrix, options)
    try:
        return solve_ksp(ksp, pc, jax_coo, b)
    finally:
        cleanup_ksp(ksp)
        cleanup_pc(pc)
        cleanup_matrix(matrix)
