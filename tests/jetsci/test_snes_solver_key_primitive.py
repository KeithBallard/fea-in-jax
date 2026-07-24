"""Smoke test for the solver-key PETSc SNES primitive path.

This checks that:
1. a PETSc SNES/KSP pair can be created and stored behind a solver key,
2. the primitive can fetch that live pair from the dictionary,
3. the primal SNES solve works, and
4. the custom JVP path uses the companion KSP for the IFT linear solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from options import *

from petsc_snes.differentiable_snes import *

from petsc_snes.solver_lifecycle import *


jax.config.update("jax_enable_x64", True)



def make_system():
    a = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )

    def residual(phi, x):
        return a @ x - jnp.sin(phi)

    def jacobian(x):
        del x
        return a

    return a, residual, jacobian



def main():
    a, residual, jacobian = make_system()
    phi = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    phi_2 = jnp.array([-3.0, 10.0, 3.0], dtype=jnp.float64)
    x0 = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)
    phi_dot = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_precond_type=PETScPreconditionerType.NONE,
        linear_solve_type=PETScLinearSolverType.LGMRES,
    )

    def primal_residual_for(active_phi):
        return jax.tree_util.Partial(lambda x: residual(active_phi, x))

    primal_jacobian = jax.tree_util.Partial(jacobian)

    # Build the live PETSc solver tuple once and store it behind a key.
    solver, options = build_petsc_solver_with_reuse(
        options,
        primal_residual_for(phi),
        primal_jacobian,
        x0,
    )
    assert options.solver_key is not None

    primitive = DifferentiableSNESPrimitive(
        residual=residual,
        solver_key=options.solver_key,
    )
    solve = make_differentiable_snes_solve(primitive)

    x_star = solve(phi, x0)
    expected_x = jnp.linalg.solve(a, phi)

    solve_with_x0 = jax.tree_util.Partial(solve, x0=x0)

    jac_fwd = jax.jacfwd(solve_with_x0)(phi)
    expected_jac = jnp.linalg.inv(a)

    _, x_dot = jax.jvp(solve_with_x0, (phi,), (phi_dot,))
    expected_x_dot = expected_jac @ phi_dot

    _, companion_ksp = get_petsc_solver_objects_from_key(options.solver_key)
    ksp_rhs = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    ksp_x = companion_ksp.solve_to_jax(ksp_rhs)
    expected_ksp_x = jnp.linalg.solve(a, ksp_rhs)

    # Now refresh the live PETSc callbacks with a different phi and verify the
    # same solver key reuses the existing SNES/KSP objects for the new problem.
    solver, options = build_petsc_solver_with_reuse(
        options,
        primal_residual_for(phi_2),
        primal_jacobian,
        x0,
    )
    x_star_2 = solve(phi_2, x0)
    expected_x_2 = jnp.linalg.solve(a, phi_2)

    print("Testing solver-key-backed SNES primitive path.")
    print("solver key:", options.solver_key)
    print("x_star:", x_star)
    print("expected x:", expected_x)
    print("x_star 2:", x_star_2)
    print("expected x 2:", expected_x_2)
    print("jacfwd:")
    print(jac_fwd)
    print("expected jac:")
    print(expected_jac)
    print("jvp x_dot:", x_dot)
    print("expected x_dot:", expected_x_dot)
    print("companion KSP solve:", ksp_x)
    print("expected companion KSP solve:", expected_ksp_x)

    np.testing.assert_allclose(np.asarray(x_star), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_star_2), np.asarray(expected_x_2), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_fwd), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_dot), np.asarray(expected_x_dot), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ksp_x), np.asarray(expected_ksp_x), rtol=1e-10, atol=1e-10)

    destroy_petsc_solver(options.solver_key)


if __name__ == "__main__":
    main()
