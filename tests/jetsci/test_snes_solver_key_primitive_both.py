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

import time

from options import *

from petsc_snes.differentiable_snes import *

from petsc_snes.solver_lifecycle import *


jax.config.update("jax_enable_x64", True)


def make_system(size: int):
    idx = jnp.arange(size, dtype=jnp.float64)
    diag = 4.0 + 0.15 * idx + 0.2 * jnp.sin(idx)
    upper = 0.5 + 0.05 * jnp.cos(idx[:-1] + 1.0)
    lower = 0.25 + 0.03 * jnp.sin(idx[:-1] + 2.0)
    a = jnp.diag(diag)
    a = a.at[jnp.arange(size - 1), jnp.arange(1, size)].set(upper)
    a = a.at[jnp.arange(1, size), jnp.arange(size - 1)].set(lower)

    def residual(phi, x):
        return a @ x - jnp.sin(phi)

    residual  = jax.jit(residual)

    def jacobian(x):
        del x
        return a
    
    jacobian = jax.jit(jacobian)

    return a, residual, jacobian

def main(size = 1000):

    a, residual, jacobian = make_system(size)
    idx = jnp.arange(size, dtype=jnp.float64)
    phi = 4.0 + 0.6 * idx + 0.2 * jnp.sin(idx + 0.25)
    phi_2 = -3.0 + 0.4 * idx + 0.35 * jnp.cos(idx + 0.5)
    x0 = 1.5 + 0.1 * idx
    phi_dot = 0.1 * jnp.sin(idx + 0.75)

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_precond_type=PETScPreconditionerType.JACOBI,
        linear_solve_type=PETScLinearSolverType.LGMRES,
    )

    def primal_residual_for(active_phi):
        return jax.tree_util.Partial(lambda x: residual(active_phi, x))

    primal_jacobian = jax.tree_util.Partial(jacobian)

    # Build the live PETSc solver tuple once and store it behind a key.
    print("Testing building solver")
    solver, options = build_petsc_solver_with_reuse(
        options,
        primal_residual_for(phi),
        primal_jacobian,
    )
    assert options.solver_key is not None
    print("solver_key",options.solver_key)

    primitive = DifferentiableSNESPrimitive(
        residual=residual,
        solver_key=options.solver_key,
    )
    print("Testing solver")
    solve = make_differentiable_snes_solve(primitive)


    start = time.time()
    x_star = solve(phi, x0)
    jax.block_until_ready(x_star)
    PETScTime = time.time()-start


    start = time.time()
    expected_x = jnp.linalg.solve(a, jnp.sin(phi))
    jax.block_until_ready(expected_x)
    JAXTime = time.time()-start

    print("Testing solver modification")
    solve = make_differentiable_snes_solve(primitive)

    solver, options = build_petsc_solver_with_reuse( #we really should split this or add some logic
        options,
        primal_residual_for(phi_2),
        primal_jacobian,
        x0,
    )

    print("modified solver_key",options.solver_key)

    x_star_2 = solve(phi_2, x0)
    expected_x_2 = jnp.linalg.solve(a, jnp.sin(phi_2))



    print("x_star:", x_star)
    print("expected x:", expected_x)
    print("x_star 2:", x_star_2)
    print("expected x 2:", expected_x_2)

    print("Testing forward differentiation")

    solve_with_x0 = jax.tree_util.Partial(solve, x0=x0)
    jac_fwd = jax.jacfwd(solve_with_x0)(phi)
    expected_jac = jnp.linalg.solve(a, jnp.diag(jnp.cos(phi)))

    print("jacfwd:")
    print(jac_fwd)
    print("expected jac:")
    print(expected_jac)

    jac_fwd_2 = jax.jacfwd(solve_with_x0)(phi_2)
    expected_jac_2 = jnp.linalg.solve(a, jnp.diag(jnp.cos(phi_2)))
    print(jac_fwd_2)
    print(expected_jac_2)

    np.testing.assert_allclose(np.asarray(x_star), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_star_2), np.asarray(expected_x_2), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_fwd), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jac_fwd_2), np.asarray(expected_jac_2), rtol=1e-10, atol=1e-10)

    print("completed forward differentiation")

    print("PETSc solve took",PETScTime)
    print("JAX solve took",JAXTime)

    exit(1)

    print("Testing backwards differentiation")    

    jac_rev = jax.jacrev(solve_with_x0)(phi)
    

    _, x_dot = jax.jvp(solve_with_x0, (phi,), (phi_dot,))
    expected_x_dot = expected_jac @ phi_dot

    _, companion_ksp = get_petsc_solver_objects_from_key(options.solver_key)
    ksp_rhs = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    ksp_x = companion_ksp.solve_to_jax(ksp_rhs)
    expected_ksp_x = jnp.linalg.solve(a, ksp_rhs)

    # Now refresh the live PETSc callbacks with a different phi and verify the
    # same solver key reuses the existing SNES/KSP objects for the new problem.



    print("Testing solver-key-backed SNES primitive path.")
    print("solver key:", options.solver_key)
    
    
    print("jvp x_dot:", x_dot)
    print("expected x_dot:", expected_x_dot)
    print("companion KSP solve:", ksp_x)
    print("expected companion KSP solve:", expected_ksp_x)


    np.testing.assert_allclose(np.asarray(jac_rev), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_dot), np.asarray(expected_x_dot), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ksp_x), np.asarray(expected_ksp_x), rtol=1e-10, atol=1e-10)

    destroy_petsc_solver(options.solver_key)


if __name__ == "__main__":
    main()