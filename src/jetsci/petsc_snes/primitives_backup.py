"""Autodiff boundary for PETSc SNES-style nonlinear solves.

This module is intentionally about the mathematical differentiation rule, not
PETSc object construction. The primal solve is supplied as a hook so it can be
backed by PETSc SNES, a buffer callback, or a pure-JAX test solve.

For a residual

    R(phi, x_star(phi)) = 0

the implicit-function theorem gives

    R_x x_dot = -R_phi phi_dot.

That is the rule implemented here. The initial guess `x0` is treated as a
solver-control input; for a converged nonlinear solve it should not contribute
to the derivative of the mathematical solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp


#these are object definitions, can we change them?

#Array = Any
#ResidualFunction = Callable[[Array, Array], Array]
#NonlinearSolve = Callable[[Array, Array], Array]
#LinearSolve = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class DifferentiableSNESPrimitive:
    """Callables used by the differentiable nonlinear-solve boundary.

    `residual(phi, x)` must be a JAX function.
    `nonlinear_solve(phi, x0)` performs the primal solve and returns `x_star`.
    `linear_solve(x_star, phi, rhs)` solves with `dR/dx` at `(phi, x_star)`.
    """

    residual: ResidualFunction
    nonlinear_solve: NonlinearSolve
    linear_solve: LinearSolve | None = None


def _dense_linear_solve_from_residual(residual: ResidualFunction) -> LinearSolve:
    """Default correctness-first linear solve for the IFT system.

    This is useful for smoke tests and small systems. Production paths should
    pass a PETSc/KSP-backed linear solve hook so `R_x` is not built densely.
    """

    def linear_solve(x_star, phi, rhs):
        jac_x = jax.jacfwd(lambda active_x: residual(phi, active_x))(x_star)
        return jnp.linalg.solve(jac_x, rhs) #this should be replaced with a 

    return linear_solve


def make_differentiable_snes_solve(primitive: DifferentiableSNESPrimitive):
    """Return `solve(phi, x0)` with an IFT custom JVP rule."""

    residual = primitive.residual
    nonlinear_solve = primitive.nonlinear_solve
    linear_solve = primitive.linear_solve or _dense_linear_solve_from_residual(residual)

    @jax.custom_jvp
    def solve(phi, x0):
        x_star = nonlinear_solve(phi, jax.lax.stop_gradient(x0))
        return jax.lax.stop_gradient(x_star)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        phi, x0 = primals
        phi_dot, _x0_dot = tangents

        x_star = solve(phi, x0)

        if type(phi_dot).__name__ == "Zero":
            return x_star, jnp.zeros_like(x_star)

        def residual_at_solution(active_phi):
            return residual(active_phi, x_star)

        _, residual_phi_dot = jax.jvp(
            residual_at_solution,
            (phi,),
            (phi_dot,),
        )
        x_dot = linear_solve(x_star, phi, -residual_phi_dot)
        return x_star, x_dot

    return solve


__all__ = [
    "DifferentiableSNESPrimitive",
    "make_differentiable_snes_solve",
]
