"""Test differentiable SNES prototype with PETSc SNES as the primal solve.

The IFT linear solve is still pure JAX in this test. That isolates whether the
prototype can use PETSc SNES for the nonlinear solve while preserving the same
JVP/Jacobian behavior validated by the pure-JAX test.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_differentiable_snes_with_petsc_primal.py
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


def residual_func(x, phi):
    return x * x - phi


def main():
    hooks = DifferentiableSNESHooks(
        nonlinear_solve=petsc_snes_solve_for_prototype,
        linear_solve=pure_jax_linear_solve_for_testing,
    )
    solve = differentiablePETScSolvePrototype(
        residual_func,
        hooks,
        options={"rtol": 1e-12, "atol": 1e-12, "max_it": 20},
    )

    phi = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    x0 = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)

    x_star = solve(phi, x0)
    expected_x = jnp.sqrt(phi)

    jac = jax.jacfwd(lambda active_phi: solve(active_phi, x0))(phi)
    expected_jac = jnp.diag(0.5 / jnp.sqrt(phi))

    phi_dot = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)
    _, x_dot = jax.jvp(lambda active_phi: solve(active_phi, x0), (phi,), (phi_dot,))
    expected_x_dot = (0.5 / jnp.sqrt(phi)) * phi_dot

    delta_phi = jnp.array([1.0e-4, -2.0e-4, 3.0e-4], dtype=jnp.float64)
    x_perturbed = solve(phi + delta_phi, x0)
    x_linearized = x_star + jac @ delta_phi
    perturbation_error = jnp.linalg.norm(x_perturbed - x_linearized)
    actual_change = x_perturbed - x_star
    predicted_change = jac @ delta_phi

    print("Testing differentiable SNES prototype with PETSc primal solve.")
    print("x_star:", x_star)
    print("expected x:", expected_x)
    print("jacfwd solve:")
    print(jac)
    print("expected jac:")
    print(expected_jac)
    print("jvp x_dot:", x_dot)
    print("expected x_dot:", expected_x_dot)
    print("delta_phi:", delta_phi)
    print("actual x change after resolving:", actual_change)
    print("linearized x change from jacobian:", predicted_change)
    print("linearization error norm:", perturbation_error)

    assert jnp.allclose(x_star, expected_x, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(jac, expected_jac, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(x_dot, expected_x_dot, rtol=1e-10, atol=1e-10)
    assert perturbation_error < 1e-8


if __name__ == "__main__":
    main()
