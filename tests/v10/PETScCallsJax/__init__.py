"""PETSc KSP -> JAX matvec method stack.

This package covers the path where PETSc owns the Krylov iteration and calls a
JAX function through a PETSc Python Mat.
"""

from .jax_mat import JaxMatContext
from .linear_methods import (
    DEFAULT_KSP_CALL_JAX_OPTIONS,
    JaxMatObjects,
    cleanup_jax_mat,
    cleanup_ksp,
    init_jax_mat,
    init_ksp_for_jax_mat,
    jax_call_petsc_solve,
    petsc_solve,
)
from .solver_call_methods import (
    KSPCallingJaxObjects,
    buildSolverObjects,
    build_solver_objects,
    cleanupSolverObjects,
    cleanup_solver_objects,
    runSimulationWithSolverObjects,
    run_simulation_with_solver_objects,
    solveWithSolverObjects,
    solve_with_solver_objects,
)

__all__ = [
    "JaxMatContext",
    "JaxMatObjects",
    "KSPCallingJaxObjects",
    "DEFAULT_KSP_CALL_JAX_OPTIONS",
    "buildSolverObjects",
    "build_solver_objects",
    "cleanupSolverObjects",
    "cleanup_jax_mat",
    "cleanup_ksp",
    "cleanup_solver_objects",
    "init_jax_mat",
    "init_ksp_for_jax_mat",
    "jax_call_petsc_solve",
    "petsc_solve",
    "runSimulationWithSolverObjects",
    "run_simulation_with_solver_objects",
    "solveWithSolverObjects",
    "solve_with_solver_objects",
]
