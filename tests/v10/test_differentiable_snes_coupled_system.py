"""Differentiable PETSc SNES prototype on a coupled nonlinear system.

This checks that the IFT Jacobian predicts changes in the converged solution
for a larger, non-diagonal residual.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_differentiable_snes_coupled_system.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.NonlinearSNES import (
    DifferentiableSNESHooks,
    differentiablePETScSolvePrototype,
    petsc_snes_solve_for_prototype,
    pure_jax_linear_solve_for_testing,
)


jax.config.update("jax_enable_x64", True)


def coupled_residual(x, phi):
    """A periodic coupled nonlinear system with phi as the forcing."""
    left = jnp.roll(x, 1)
    right = jnp.roll(x, -1)
    return (
        x
        + 0.08 * x**3
        + 0.12 * (2.0 * x - left - right)
        + 0.02 * x * right
        - phi
    )


def main():
    hooks = DifferentiableSNESHooks(
        nonlinear_solve=petsc_snes_solve_for_prototype,
        linear_solve=pure_jax_linear_solve_for_testing,
    )
    solve = differentiablePETScSolvePrototype(
        coupled_residual,
        hooks,
        options={"rtol": 1e-12, "atol": 1e-12, "max_it": 40},
    )

    n = 100
    grid = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False, dtype=jnp.float64)
    phi = 0.2 + 0.05 * jnp.sin(grid) + 0.03 * jnp.cos(3.0 * grid)
    x0 = phi

    x_star = solve(phi, x0)
    residual_norm = jnp.linalg.norm(coupled_residual(x_star, phi))

    jac = jax.jacfwd(lambda active_phi: solve(active_phi, x0))(phi)

    delta_phi = 1.0e-5 * (0.7 * jnp.sin(2.0 * grid) - 0.4 * jnp.cos(5.0 * grid))
    x_perturbed = solve(phi + delta_phi, x0)
    actual_change = x_perturbed - x_star
    predicted_change = jac @ delta_phi
    linearization_error = jnp.linalg.norm(actual_change - predicted_change)
    actual_change_norm = jnp.linalg.norm(actual_change)
    relative_error = linearization_error / jnp.maximum(actual_change_norm, 1.0e-30)

    print("Testing differentiable PETSc SNES prototype on coupled nonlinear system.")
    print("system size:", n)
    print("residual norm at x_star:", residual_norm)
    print("x_star first 8:", x_star[:8])
    print("jacobian shape:", jac.shape)
    print("delta_phi first 8:", delta_phi[:8])
    print("actual x change first 8:", actual_change[:8])
    print("predicted x change first 8:", predicted_change[:8])
    print("linearization error norm:", linearization_error)
    print("actual change norm:", actual_change_norm)
    print("relative linearization error:", relative_error)

    assert residual_norm < 1e-9
    assert relative_error < 1e-3


if __name__ == "__main__":
    main()
