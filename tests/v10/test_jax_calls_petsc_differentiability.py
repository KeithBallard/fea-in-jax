"""Differentiability smoke test for the v9 JAX -> PETSc KSP path.

Run from the PETScJVP root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_jax_calls_petsc_differentiability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.JaxCallsPETSc import (
    COOData,
    PETScKSPType,
    PETScMethodOptions,
    PETScPCType,
    buildSolverObjects,
    cleanupSolverObjects,
    solveWithSolverObjects,
)


jax.config.update("jax_enable_x64", True)


def diagonal_coo_data(diag):
    n = diag.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)
    return COOData(
        shape=jnp.array([n, n], dtype=jnp.int64),
        vals=diag,
        rows=indices,
        cols=indices,
    )


def diagonal_matrix_function(diag):
    def matrix_function(_x):
        return diagonal_coo_data(diag)

    return matrix_function


def print_array(label, value):
    print(f"{label}: {np.asarray(value)}")


def main():
    diag = jnp.array([4.0, 3.0, 9.0, 3.0, 4.0, 8.0, 6.0, 4.0], dtype=jnp.float64)
    b = jnp.ones_like(diag)
    x0 = jnp.zeros_like(diag)
    expected_x = b / diag
    expected_grad_vals = -b / (diag * diag)
    expected_jac_b = jnp.diag(1.0 / diag)

    options = PETScMethodOptions(
        ksp_type=PETScKSPType.CG,
        pc_type=PETScPCType.JACOBI,
    )

    solver_objects = buildSolverObjects(diagonal_matrix_function(diag), x0, options)
    try:
        def solve(vals, rhs):
            return solveWithSolverObjects(
                solver_objects,
                rhs,
                matrix_data=diagonal_coo_data(vals),
            )

        print("Testing primal solve used by differentiability checks.")
        x = solve(diag, b)
        x.block_until_ready()
        print_array("x", x)
        print_array("expected x", expected_x)
        np.testing.assert_allclose(np.asarray(x), np.asarray(expected_x), rtol=1e-10, atol=1e-10)

        print("Testing grad wrt visible matrix diagonal values.")
        grad_vals = jax.grad(lambda vals: jnp.sum(solve(vals, b)))(diag)
        print_array("grad vals", grad_vals)
        print_array("expected grad vals", expected_grad_vals)
        np.testing.assert_allclose(
            np.asarray(grad_vals),
            np.asarray(expected_grad_vals),
            rtol=1e-10,
            atol=1e-10,
        )

        print("Testing jvp wrt both matrix values and RHS.")
        vals_dot = jnp.full_like(diag, 0.1)
        b_dot = jnp.full_like(b, 0.2)
        x_primal, x_tangent = jax.jvp(
            solve,
            (diag, b),
            (vals_dot, b_dot),
        )
        expected_tangent = (b_dot - vals_dot * expected_x) / diag
        print_array("jvp primal", x_primal)
        print_array("jvp tangent", x_tangent)
        print_array("expected tangent", expected_tangent)
        np.testing.assert_allclose(np.asarray(x_primal), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(x_tangent),
            np.asarray(expected_tangent),
            rtol=1e-10,
            atol=1e-10,
        )

        print("Testing jacfwd wrt RHS.")
        jac_b = jax.jacfwd(lambda rhs: solve(diag, rhs))(b)
        print_array("jacfwd wrt b", jac_b)
        print_array("expected jacfwd wrt b", expected_jac_b)
        np.testing.assert_allclose(np.asarray(jac_b), np.asarray(expected_jac_b), rtol=1e-10, atol=1e-10)

        print("Testing direct vjp wrt matrix values.")
        x_vjp, pullback = jax.vjp(lambda vals: solve(vals, b), diag)
        vals_bar = pullback(jnp.ones_like(x_vjp))[0]
        print_array("vjp primal", x_vjp)
        print_array("vjp vals_bar", vals_bar)
        print_array("expected vals_bar", expected_grad_vals)
        np.testing.assert_allclose(np.asarray(x_vjp), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(vals_bar),
            np.asarray(expected_grad_vals),
            rtol=1e-10,
            atol=1e-10,
        )
    finally:
        cleanupSolverObjects(solver_objects)


if __name__ == "__main__":
    main()
