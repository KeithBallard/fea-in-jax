"""Assembled sparse matrix PETSc solve path for the bridge prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp

from .linearization import LinearSystemObjects
from .options import PETScKSPOptions


def _add_ksp_jax_rules_to_path():
    package_dir = Path(__file__).resolve().parent
    candidates = [
        # Target fea-in-jax layout:
        #   src/fe_jax/petsc_backend/ksp_jax_rules/
        package_dir / "ksp_jax_rules",
        # Temporary layout if the v7 folder name is copied over literally:
        #   src/fe_jax/petsc_backend/KSP-Jax-rules/
        package_dir / "KSP-Jax-rules",
        # Current PETScJVP v7 prototype layout:
        #   v7/KSP-Jax-rules/
        package_dir.parents[0] / "KSP-Jax-rules",
    ]

    for candidate in candidates:
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return

    raise ModuleNotFoundError(
        "Could not find KSP-Jax-rules. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


_add_ksp_jax_rules_to_path()

import buildKSP_Keith  # noqa: E402
import differentiateKSP_Keith  # noqa: E402


@dataclass
class PETScPersistentLinearState:
    """Persistent PETSc state for fixed-sparsity nonlinear iterations."""

    matrix: Any
    pc: Any
    solver: Any
    shape: jax.Array
    rows: jax.Array
    cols: jax.Array
    options: PETScKSPOptions


def coo_parts(jax_coo) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the tuple convention used by v7: shape, vals, rows, cols."""
    return (
        jnp.array(jax_coo.shape, dtype=jnp.int64),
        jax_coo.data,
        jnp.asarray(jax_coo.row, dtype=jnp.int32),
        jnp.asarray(jax_coo.col, dtype=jnp.int32),
    )


def solve_with_rebuild(jax_coo, rhs, options: PETScKSPOptions = PETScKSPOptions()):
    """Build Mat/PC/KSP, solve once, and cleanup."""
    shape, vals, rows, cols = coo_parts(jax_coo)

    matrix = buildKSP_Keith.linearMatrixInit(
        jac=lambda _x0: (shape, vals, rows, cols),
        x0=None,
        constructionOptions=options.as_matrix_construction_options(),
    )
    pc = buildKSP_Keith.linearPCInit(
        matrix,
        constructionOptions=options.as_pc_construction_options(),
    )
    solver = buildKSP_Keith.linearKSPInit(
        matrix,
        constructionOptions=options.as_ksp_construction_options(),
    )

    try:
        return differentiateKSP_Keith.linearSolverSolve(
            solver,
            pc,
            shape,
            rows,
            cols,
            vals,
            rhs,
        )
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


def init_persistent_state(
    jax_coo,
    options: PETScKSPOptions = PETScKSPOptions(),
) -> PETScPersistentLinearState:
    """Create persistent PETSc Mat/PC/KSP for a fixed sparsity pattern."""
    shape, vals, rows, cols = coo_parts(jax_coo)

    matrix = buildKSP_Keith.linearMatrixInit(
        jac=lambda _x0: (shape, vals, rows, cols),
        x0=None,
        constructionOptions=options.as_matrix_construction_options(),
    )
    pc = buildKSP_Keith.linearPCInit(
        matrix,
        constructionOptions=options.as_pc_construction_options(),
    )
    solver = buildKSP_Keith.linearKSPInit(
        matrix,
        constructionOptions=options.as_ksp_construction_options(),
    )

    return PETScPersistentLinearState(
        matrix=matrix,
        pc=pc,
        solver=solver,
        shape=shape,
        rows=rows,
        cols=cols,
        options=options,
    )


def solve_with_persistent_state(
    state: PETScPersistentLinearState,
    jax_coo,
    rhs,
    *,
    update_matrix: bool = True,
):
    """Update values in a persistent PETSc Mat, then solve."""
    shape, vals, rows, cols = coo_parts(jax_coo)

    if update_matrix:
        # This return value is a sequencing token, not a new matrix object.
        # The PETSc Mat is mutated in-place behind the persistent handle, but
        # the subsequent solve only receives the KSP/PC handles. Until the solve
        # primitive has an explicit dependency on the matrix-update primitive,
        # force this token ready so JAX cannot drop or reorder the update.
        matrix_update_token = buildKSP_Keith.linearMatrixUpdate(state.matrix, vals)
        matrix_update_token.handle.block_until_ready()

    return differentiateKSP_Keith.linearSolverSolve(
        state.solver,
        state.pc,
        shape,
        rows,
        cols,
        vals,
        rhs,
    )


def solve_linear_system_objects(
    system: LinearSystemObjects,
    *,
    persistent_state: PETScPersistentLinearState | None = None,
    options: PETScKSPOptions = PETScKSPOptions(),
):
    """Solve from evaluated JAX matrix/vector objects."""
    if system.J_sparse is None:
        raise ValueError("Assembled PETSc solve requires an evaluated sparse Jacobian.")

    if persistent_state is None:
        return solve_with_rebuild(system.J_sparse, system.rhs, options)

    return solve_with_persistent_state(persistent_state, system.J_sparse, system.rhs)


def cleanup_persistent_state(state: PETScPersistentLinearState):
    """Destroy persistent PETSc objects."""
    buildKSP_Keith.linearSolverCleanup(state.solver)
    buildKSP_Keith.linearPCCleanup(state.pc)
    buildKSP_Keith.linearMatrixCleanup(state.matrix)
