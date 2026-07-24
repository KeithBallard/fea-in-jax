import jax
import jax.numpy as jnp

from options import *
from solve import *

jax.config.update("jax_enable_x64", True)

import numpy as np


def matrix_from_phi(phi):
    phi = jnp.asarray(phi, dtype=jnp.float64)
    n = phi.shape[0]
    diag = 4.0 + 0.10 * phi
    A = jnp.diag(diag)
    A = A + 0.20 * jnp.diag(jnp.ones(n - 1, dtype=jnp.float64), k=1)
    A = A + 0.15 * jnp.diag(jnp.ones(n - 1, dtype=jnp.float64), k=-1)
    return A


def residual(phi, x):
    return matrix_from_phi(phi) @ x - phi


def solve_with_petsc(phi, x0, options):
    x, _ = differentiable_solve(
        options,
        residual,
        None,
        x0,
        phi,
    )
    return x


def main():
    n = 50
    phi = jnp.linspace(1.0, 5.0, n, dtype=jnp.float64)
    phi_2 = jnp.linspace(-2.0, 8.0, n, dtype=jnp.float64)
    x0 = jnp.linspace(0.5, 2.5, n, dtype=jnp.float64)

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_precond_type=PETScPreconditionerType.JACOBI,
        linear_solve_type=PETScLinearSolverType.LGMRES,
    )

    print("Testing PETSc SNES on a phi-dependent nonlinear residual.")
    print("n:", n)
    print("phi first 8:", phi[:8])
    print("phi_2 first 8:", phi_2[:8])

    sol = solve_with_petsc(phi, x0, options)
    sol_2 = solve_with_petsc(phi_2, x0, options)

    expected_x = jnp.linalg.solve(matrix_from_phi(phi), phi)
    expected_x_2 = jnp.linalg.solve(matrix_from_phi(phi_2), phi_2)

    print("PETSc sol(phi) first 8:", sol[:8])
    print("expected sol(phi) first 8:", expected_x[:8])
    print("PETSc sol(phi_2) first 8:", sol_2[:8])
    print("expected sol(phi_2) first 8:", expected_x_2[:8])
    print("Residual(phi, sol(phi)) norm:", jnp.linalg.norm(residual(phi, sol)))
    print("Residual(phi_2, sol(phi_2)) norm:", jnp.linalg.norm(residual(phi_2, sol_2)))

    np.testing.assert_allclose(np.asarray(sol), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(sol_2), np.asarray(expected_x_2), rtol=1e-10, atol=1e-10)
    assert not np.allclose(np.asarray(sol), np.asarray(sol_2))


    print("Attempting PETSc-backed differentiation wrt phi.")
    solve_only = lambda active_phi: solve_with_petsc(active_phi, x0, options)
    reference_solve = lambda active_phi: jnp.linalg.solve(matrix_from_phi(active_phi), active_phi)
    expected_jac = jax.jacfwd(reference_solve)(phi)
    expected_jac_2 = jax.jacfwd(reference_solve)(phi_2)

    jac_fwd = jax.jacfwd(solve_only)(phi)
    jac_rev = jax.jacrev(solve_only)(phi)
    jac_fwd_2 = jax.jacfwd(solve_only)(phi_2)

    print("PETSc jacfwd(phi) first 6x6:")
    print(jac_fwd[:6, :6])
    print("PETSc jacrev(phi) first 6x6:")
    print(jac_rev[:6, :6])
    print("Expected jacfwd(phi) first 6x6:")
    print(expected_jac[:6, :6])
    print("PETSc jacfwd(phi_2) first 6x6:")
    print(jac_fwd_2[:6, :6])
    print("Expected jacfwd(phi_2) first 6x6:")
    print(expected_jac_2[:6, :6])

    np.testing.assert_allclose(np.asarray(jac_fwd), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_rev), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_fwd_2), np.asarray(expected_jac_2), rtol=1e-10, atol=1e-10)
    assert not np.allclose(np.asarray(jac_fwd), np.asarray(jac_fwd_2))


if __name__ == "__main__":
    main()
