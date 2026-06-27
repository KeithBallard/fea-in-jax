"""Prototype bridge between Nathan's cleaned-up linear_solve and v7 PETSc KSP.

Conceptual placement in nathan/contact:

    src/fe_jax/sparse_linear_solve.py
        computes R_0 and J_w_dirichlet(x_0)
        calls this backend for LinearSolverType.PETSC

    src/fe_jax/petsc_backend/
        owns a cleaned-up version of this package plus the v7 KSP-Jax modules

The v7 `__CupyCtx` objects used by the lower layers are PETSc object handles.
They are named after CuPy because the scalar handle is transported through
buffer_callback buffers with CuPy, not because the stored object is a CuPy
solver.
"""

from .assembled import (
    PETScPersistentLinearState,
    cleanup_persistent_state,
    coo_parts,
    init_persistent_state,
    solve_linear_system_objects,
    solve_with_persistent_state,
    solve_with_rebuild,
)
from .entrypoints import (
    nathan_callable_linear_solve_petsc_branch,
    nathan_linear_solve_petsc_branch,
    solve_evaluated_petsc_branch,
)
from .linearization import (
    Jacobian,
    JacobianDiagonl,
    JacobianDiagonal,
    LinearizationCallables,
    LinearSystemObjects,
    Residual,
    as_jacobian,
    as_jacobian_diagonal,
    as_residual,
    build_linear_system_objects,
    build_linearization_callables,
    callables_to_linear_system_objects,
    evaluate_linearization_callables,
    jacobian_diagonal_to_callable,
    jacobian_to_callable,
    jvp_callable_from_residual,
    residual_to_callable,
)
from .options import PETScKSPOptions, PETScKSPType, PETScMatrixType, PETScPCType


__all__ = [
    "Jacobian",
    "JacobianDiagonal",
    "JacobianDiagonl",
    "LinearizationCallables",
    "LinearSystemObjects",
    "PETScKSPOptions",
    "PETScKSPType",
    "PETScMatrixType",
    "PETScPCType",
    "PETScPersistentLinearState",
    "Residual",
    "as_jacobian",
    "as_jacobian_diagonal",
    "as_residual",
    "build_linear_system_objects",
    "build_linearization_callables",
    "callables_to_linear_system_objects",
    "cleanup_persistent_state",
    "coo_parts",
    "evaluate_linearization_callables",
    "init_persistent_state",
    "jacobian_diagonal_to_callable",
    "jacobian_to_callable",
    "jvp_callable_from_residual",
    "nathan_callable_linear_solve_petsc_branch",
    "nathan_linear_solve_petsc_branch",
    "residual_to_callable",
    "solve_evaluated_petsc_branch",
    "solve_linear_system_objects",
    "solve_with_persistent_state",
    "solve_with_rebuild",
]
