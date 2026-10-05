"""Smoke test for the jetsci SNES differentiation primitive.

This is intentionally loaded directly from the module file so it does not
depend on the package import wiring, which is still being cleaned up.
The shape mirrors the v11 primitive test and is meant to be easy to sprinkle
with debug prints while we inspect what the primitive forwards into the
linear solve path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMITIVE_PATH = REPO_ROOT / "jetsci" / "petsc_snes" / "primitives.py"

spec = importlib.util.spec_from_file_location("jetsci_petsc_snes_primitives", PRIMITIVE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load primitive module from {PRIMITIVE_PATH}")
primitive_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = primitive_mod
spec.loader.exec_module(primitive_mod)


jax.config.update("jax_enable_x64", True)


def residual(phi, x):
    a = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    return a @ x - phi


def pure_jax_newton_solve(phi, x0):
    del x0
    a = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    return jnp.linalg.solve(a, phi)


def main():
    primitive = primitive_mod.DifferentiableSNESPrimitive(
        residual=residual,
        nonlinear_solve=pure_jax_newton_solve,
    )
    solve = primitive_mod.make_differentiable_snes_solve(primitive)

    phi = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    x0 = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)
    phi_dot = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)

    x_star = solve(phi, x0)
    a = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    expected_x = jnp.linalg.solve(a, phi)
    expected_jac = jnp.linalg.inv(a)

    jac_fwd = jax.jacfwd(lambda active_phi: solve(active_phi, x0))(phi)
    jac_rev = jax.jacrev(lambda active_phi: solve(active_phi, x0))(phi)
    _, x_dot = jax.jvp(lambda active_phi: solve(active_phi, x0), (phi,), (phi_dot,))
    grad_sum = jax.grad(lambda active_phi: jnp.sum(solve(active_phi, x0)))(phi)

    print("Testing jetsci differentiable SNES primitive boundary.")
    print("x_star:", x_star)
    print("expected x:", expected_x)
    print("jacfwd:")
    print(jac_fwd)
    print("jacrev:")
    print(jac_rev)
    print("expected jac:")
    print(expected_jac)
    print("jvp x_dot:", x_dot)
    print("expected x_dot:", expected_jac @ phi_dot)
    print("grad sum:", grad_sum)
    print("expected grad sum:", jnp.sum(expected_jac, axis=0))

    np.testing.assert_allclose(np.asarray(x_star), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_fwd), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_rev), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_dot), np.asarray(expected_jac @ phi_dot), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(grad_sum), np.asarray(jnp.sum(expected_jac, axis=0)), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    main()
