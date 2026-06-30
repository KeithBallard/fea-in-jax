"""Nathan/contact-facing PETSc bridge entry points."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.experimental.sparse as jsparse

from .assembled import (
    PETScPersistentLinearState,
    solve_linear_system_objects,
    solve_with_persistent_state,
    solve_with_rebuild,
)
from .linearization import (
    Jacobian,
    JacobianDiagonal,
    LinearSystemObjects,
    Residual,
    build_linear_system_objects,
    callables_to_linear_system_objects,
)
from .options import PETScKSPOptions


def nathan_linear_solve_petsc_branch(
    *,
    J_w_dirichlet: Callable[[jax.Array], Any],
    x_0: jax.Array,
    R_0: jax.Array,
    constraints: Any,
    persistent_state: Optional[PETScPersistentLinearState] = None,
    options: PETScKSPOptions = PETScKSPOptions(),
):
    """Prototype body for a future LinearSolverType.PETSC branch."""
    system = callables_to_linear_system_objects(
        R_w_dirichlet=lambda _x: R_0,
        J_w_dirichlet=J_w_dirichlet,
        x_0=x_0,
    )
    delta_x = solve_linear_system_objects(
        system,
        persistent_state=persistent_state,
        options=options,
    )
    return constraints.apply_to_delta_solution(delta_x, x_0)


def nathan_callable_linear_solve_petsc_branch(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonal],
    constraints: Any,
    x_0: jax.Array,
    *args,
    persistent_state: Optional[PETScPersistentLinearState] = None,
    options: PETScKSPOptions = PETScKSPOptions(),
    apply_dirichlet_lhs_fn: Optional[Callable[[jsparse.COO, jax.Array], jsparse.COO]] = None,
    apply_dirichlet_residual_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    **kwargs,
):
    """Start from Nathan-style wrappers, evaluate them, then solve with PETSc."""
    system = build_linear_system_objects(
        residual,
        jacobian,
        jacobian_diagonal,
        constraints,
        x_0,
        *args,
        apply_dirichlet_lhs_fn=apply_dirichlet_lhs_fn,
        apply_dirichlet_residual_fn=apply_dirichlet_residual_fn,
        **kwargs,
    )
    delta_x = solve_linear_system_objects(
        system,
        persistent_state=persistent_state,
        options=options,
    )
    return constraints.apply_to_delta_solution(delta_x, x_0)


def solve_evaluated_petsc_branch(
    system: LinearSystemObjects,
    constraints: Any,
    x_0: jax.Array,
    *,
    persistent_state: Optional[PETScPersistentLinearState] = None,
    options: PETScKSPOptions = PETScKSPOptions(),
):
    """Solve after the caller has already evaluated callables into JAX objects."""
    delta_x = solve_linear_system_objects(
        system,
        persistent_state=persistent_state,
        options=options,
    )
    return constraints.apply_to_delta_solution(delta_x, x_0)
