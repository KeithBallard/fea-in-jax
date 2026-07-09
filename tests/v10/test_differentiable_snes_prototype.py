"""Smoke test for the v10 differentiable SNES prototype.

This test uses pure-JAX hooks so the IFT contract can be checked without PETSc.

Run:

    python v10/test_differentiable_snes_prototype.py
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
    pure_jax_linear_solve_for_testing,
    pure_jax_newton_solve_for_testing,
)


jax.config.update("jax_enable_x64", True)


def residual_func(x, phi):
    return x * x - phi


def main():
    hooks = DifferentiableSNESHooks(
        nonlinear_solve=pure_jax_newton_solve_for_testing,
        linear_solve=pure_jax_linear_solve_for_testing,
    )
    solve = differentiablePETScSolvePrototype(residual_func, hooks)

    phi = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    x0 = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)

    x_star = solve(phi, x0)
    expected_x = jnp.sqrt(phi)

    jac = jax.jacfwd(lambda active_phi: solve(active_phi, x0))(phi)
    expected_jac = jnp.diag(0.5 / jnp.sqrt(phi))

    phi_dot = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)
    _, x_dot = jax.jvp(lambda active_phi: solve(active_phi, x0), (phi,), (phi_dot,))
    expected_x_dot = (0.5 / jnp.sqrt(phi)) * phi_dot

    print("Testing differentiable SNES prototype with pure-JAX hooks.")
    print("x_star:", x_star)
    print("expected x:", expected_x)
    print("jacfwd solve:")
    print(jac)
    print("expected jac:")
    print(expected_jac)
    print("jvp x_dot:", x_dot)
    print("expected x_dot:", expected_x_dot)

    assert jnp.allclose(x_star, expected_x, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(jac, expected_jac, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(x_dot, expected_x_dot, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    main()
