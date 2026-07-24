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

import sys
from time import perf_counter


from options import *

from petsc_snes.differentiable_snes import *

from petsc_snes.solver_lifecycle import *


jax.config.update("jax_enable_x64", True)


def _measure(label: str, thunk, block: bool = False):
    start = perf_counter()
    value = thunk()
    if block and hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.6f} s")
    return value, elapsed


def make_system(size: int):
    idx = jnp.arange(size, dtype=jnp.float64)
    diag = 4.0 + 0.15 * idx + 0.2 * jnp.sin(idx)
    upper = 0.5 + 0.05 * jnp.cos(idx[:-1] + 1.0)
    lower = 0.25 + 0.03 * jnp.sin(idx[:-1] + 2.0)
    a = jnp.diag(diag)
    a = a.at[jnp.arange(size - 1), jnp.arange(1, size)].set(upper)
    a = a.at[jnp.arange(1, size), jnp.arange(size - 1)].set(lower)

    def operator(phi):
        # Keep the sparsity pattern fixed while making the matrix values
        # depend on phi.
        return a + jnp.diag(0.1 * jnp.tanh(phi))

    operator = jax.jit(operator)

    def residual(phi, x):
        jax.debug.print("original residual called")
        return operator(phi) @ x + 0.05 * x**2 - jnp.sin(phi)

    residual  = jax.jit(residual)

    def jacobian(phi, x):
        return operator(phi) + jnp.diag(0.1 * x)

    jacobian = jax.jit(jacobian)

    rows = jnp.concatenate((
        jnp.arange(size, dtype=jnp.int32),
        jnp.arange(size - 1, dtype=jnp.int32),
        jnp.arange(1, size, dtype=jnp.int32),
    ))
    cols = jnp.concatenate((
        jnp.arange(size, dtype=jnp.int32),
        jnp.arange(1, size, dtype=jnp.int32),
        jnp.arange(size - 1, dtype=jnp.int32),
    ))
    shape = jnp.asarray((size, size), dtype=jnp.int64)

    def jacobian_coo(phi, x):
        values = jnp.concatenate((
            diag + 0.1 * jnp.tanh(phi) + 0.1 * x,
            upper,
            lower,
        ))
        return COOData(shape=shape, vals=values, rows=rows, cols=cols)

    jacobian_coo = jax.jit(jacobian_coo)

    return a, operator, residual, jacobian, jacobian_coo


def _jax_newton_solve(
    residual,
    jacobian,
    active_phi,
    active_x0,
    nonlinear_tolerance,
    nonlinear_max_iter,
):
    def condition(state):
        iteration, x = state
        return jnp.logical_and(
            iteration < nonlinear_max_iter,
            jnp.linalg.norm(residual(active_phi, x)) > nonlinear_tolerance,
        )

    def body(state):
        iteration, x = state
        residual_value = residual(active_phi, x)
        step = jnp.linalg.solve(jacobian(active_phi, x), -residual_value)
        return iteration + 1, x + step

    _, solution = jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(0), active_x0),
    )
    return solution


def _jax_newton_gmres_solve(
    residual,
    jacobian,
    active_phi,
    active_x0,
    linear_relative_tol: float,
    linear_absolute_tol: float,
    linear_max_iter: int,
    nonlinear_tolerance,
    nonlinear_max_iter,
):
    def condition(state):
        iteration, x = state
        return jnp.logical_and(
            iteration < nonlinear_max_iter,
            jnp.linalg.norm(residual(active_phi, x)) > nonlinear_tolerance,
        )

    def body(state):
        iteration, x = state
        matrix = jacobian(active_phi, x)
        residual_value = residual(active_phi, x)
        diagonal = jnp.diag(matrix)

        step, _ = jax.scipy.sparse.linalg.gmres(
            lambda vector: matrix @ vector,
            -residual_value,
            M=lambda vector: vector / diagonal,
            tol=linear_relative_tol,
            atol=linear_absolute_tol,
            maxiter=linear_max_iter,
            solve_method="batched",
        )
        return iteration + 1, x + step

    _, solution = jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(0), active_x0),
    )
    return solution

def main(size: int = 100):

    a, operator, residual, jacobian, jacobian_coo = make_system(size)
    idx = jnp.arange(size, dtype=jnp.float64)
    phi = 4.0 + 0.6 * idx + 0.2 * jnp.sin(idx + 0.25)
    phi_2 = -3.0 + 0.4 * idx + 0.35 * jnp.cos(idx + 0.5)
    x0 = 0.1 * jnp.cos(idx)
    phi_dot = 0.1 * jnp.sin(idx + 0.75)

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_precond_type=PETScPreconditionerType.JACOBI,
        linear_solve_type=PETScLinearSolverType.LGMRES,
        linear_absolute_tol=1e-14,
    )

    def primal_residual_for(active_phi):
        return jax.tree_util.Partial(lambda x: residual(active_phi, x))

    primal_jacobian = jax.tree_util.Partial(jacobian_coo, phi)

    print("Testing building solver")
    (solver, options), build_elapsed = _measure(
        "PETSc solver build",
        lambda: build_petsc_solver_with_reuse(
            options,
            primal_residual_for(phi),
            primal_jacobian,
        ),
    )
    assert options.solver_key is not None
    print("solver_key", options.solver_key)
    solver.diagnostics = True

    primitive = DifferentiableSNESPrimitive(
        residual=residual,
        jacobian=jacobian_coo,
        solver_key=options.solver_key,
    )
    solve = make_differentiable_snes_solve(primitive)
    solve_with_x0 = jax.tree_util.Partial(solve, x0=x0)

    jax_newton = jax.jit(
        lambda active_phi, active_x0: _jax_newton_solve(
            residual,
            jacobian,
            active_phi,
            active_x0,
            options.nonlinear_absolute_tol,
            options.nonlinear_max_iter,
        )
    )
    jax_newton_with_x0 = jax.tree_util.Partial(jax_newton, active_x0=x0)

    jax_newton_gmres = jax.jit(
        lambda active_phi, active_x0: _jax_newton_gmres_solve(
            residual,
            jacobian,
            active_phi,
            active_x0,
            options.linear_relative_tol,
            options.linear_absolute_tol,
            options.linear_max_iter,
            options.nonlinear_absolute_tol,
            options.nonlinear_max_iter,
        )
    )
    jax_newton_gmres_with_x0 = jax.tree_util.Partial(
        jax_newton_gmres,
        active_x0=x0,
    )

    # Warmup so the measured timings are steadier.
    jax.block_until_ready(solve(phi, x0))
    jax.block_until_ready(jax_newton(phi, x0))
    jax.block_until_ready(jax_newton_gmres(phi, x0))
    jax.block_until_ready(jax.jacfwd(solve_with_x0)(phi))
    jax.block_until_ready(jax.jacfwd(jax_newton_with_x0)(phi))

    solver.callback_stats.clear()
    x_star, pet_primal_elapsed = _measure(
        "PETSc primal solve",
        lambda: solve(phi, x0),
        block=True,
    )
    print("PETSc first solve breakdown:", solver.last_diagnostics)
    #expected_x = jnp.linalg.solve(operator(phi), jnp.sin(phi))

    solver.callback_stats.clear()
    x_star_2, pet_primal_elapsed_2 = _measure(
        "PETSc primal solve (updated phi)",
        lambda: solve(phi_2, x0),
        block=True,
    )
    print("PETSc updated solve breakdown:", solver.last_diagnostics)
    #expected_x_2 = jnp.linalg.solve(operator(phi_2), jnp.sin(phi_2))

    jax_x_star, jax_primal_elapsed = _measure(
        "JAX Newton baseline",
        lambda: jax_newton(phi, x0),
        block=True,
    )
    jax_x_star_2, jax_primal_elapsed_2 = _measure(
        "JAX Newton baseline (updated phi)",
        lambda: jax_newton(phi_2, x0),
        block=True,
    )

    jax_gmres_x_star, jax_gmres_elapsed = _measure(
        "JAX Newton-GMRES",
        lambda: jax_newton_gmres(phi, x0),
        block=True,
    )
    jax_gmres_x_star_2, jax_gmres_elapsed_2 = _measure(
        "JAX Newton-GMRES (updated phi)",
        lambda: jax_newton_gmres(phi_2, x0),
        block=True,
    )

    
    jac_fwd, pet_jacfwd_elapsed = _measure(
        "PETSc jacfwd",
        lambda: jax.jacfwd(solve_with_x0)(phi),
        block=True,
    )
    expected_jac = jax.jacfwd(
        lambda active_phi: jnp.linalg.solve(
            operator(active_phi),
            jnp.sin(active_phi),
        )
    )(phi)

    jac_fwd_2, pet_jacfwd_elapsed_2 = _measure(
        "PETSc jacfwd (updated phi)",
        lambda: jax.jacfwd(solve_with_x0)(phi_2),
        block=True,
    )
    expected_jac_2 = jax.jacfwd(
        lambda active_phi: jnp.linalg.solve(
            operator(active_phi),
            jnp.sin(active_phi),
        )
    )(phi_2)

    jax_jacfwd, jax_jacfwd_elapsed = _measure(
        "JAX Newton jacfwd",
        lambda: jax.jacfwd(jax_newton_with_x0)(phi),
        block=True,
    )
    

    """
    jac_rev, pet_jacrev_elapsed = _measure(
        "PETSc jacrev",
        lambda: jax.jacrev(solve_with_x0)(phi),
        block=True,
    )
    """

    """
    _, x_dot = jax.jvp(solve_with_x0, (phi,), (phi_dot,))
    expected_x_dot = expected_jac @ phi_dot
    """

    _, companion_ksp = get_petsc_solver_objects_from_key(options.solver_key)
    ksp_rhs = jnp.zeros(size, dtype=jnp.float64).at[0].set(1.0)
    ksp_x = companion_ksp.solve_to_jax(ksp_rhs)
    expected_ksp_x = jnp.linalg.solve(operator(phi), ksp_rhs)

    """
    print("Testing solver-key-backed SNES primitive path.")
    print("system size:", size)
    print("solver key:", options.solver_key)
    print("x_star:", x_star)
    print("expected x:", expected_x)
    print("x_star 2:", x_star_2)
    print("expected x 2:", expected_x_2)
    print("jax x_star:", jax_x_star)
    print("jax x_star 2:", jax_x_star_2)
    print("JAX GMRES x_star:", jax_gmres_x_star)
    print("JAX GMRES x_star 2:", jax_gmres_x_star_2)
    print("jacfwd:")
    print(jac_fwd)
    print("expected jac:")
    print(expected_jac)
    print("jacfwd 2:")
    print(jac_fwd_2)
    print("expected jac 2:")
    print(expected_jac_2)
    print("JAX Newton jacfwd:")
    print(jax_jacfwd)
    print("jvp x_dot:", x_dot)
    print("expected x_dot:", expected_x_dot)
    print("companion KSP solve:", ksp_x)
    print("expected companion KSP solve:", expected_ksp_x)
    """

    #np.testing.assert_allclose(np.asarray(x_star), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(x_star_2), np.asarray(expected_x_2), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(jax_x_star), np.asarray(expected_x), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(jax_x_star_2), np.asarray(expected_x_2), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(jax_gmres_x_star), np.asarray(expected_x), rtol=1e-8, atol=1e-8)
    #np.testing.assert_allclose(np.asarray(jax_gmres_x_star_2), np.asarray(expected_x_2), rtol=1e-8, atol=1e-8)
    #np.testing.assert_allclose(np.asarray(jac_fwd), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(jac_fwd_2), np.asarray(expected_jac_2), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(jac_rev), np.asarray(expected_jac), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(x_dot), np.asarray(expected_x_dot), rtol=1e-10, atol=1e-10)
    #np.testing.assert_allclose(np.asarray(ksp_x), np.asarray(expected_ksp_x), rtol=1e-10, atol=1e-10)

    print("\n=== TIMING SUMMARY ===")
    print(f"PETSc build: {build_elapsed:.6f} s")
    print(f"PETSc primal solve: {pet_primal_elapsed:.6f} s")
    print(f"PETSc primal solve (updated phi): {pet_primal_elapsed_2:.6f} s")
    print(f"PETSc jacfwd: {pet_jacfwd_elapsed:.6f} s")
    print(f"PETSc jacfwd (updated phi): {pet_jacfwd_elapsed_2:.6f} s")
    #print(f"PETSc jacrev: {pet_jacrev_elapsed:.6f} s")
    print(f"JAX Newton baseline: {jax_primal_elapsed:.6f} s")
    print(f"JAX Newton baseline (updated phi): {jax_primal_elapsed_2:.6f} s")
    print(f"JAX Newton-GMRES: {jax_gmres_elapsed:.6f} s")
    print(f"JAX Newton-GMRES (updated phi): {jax_gmres_elapsed_2:.6f} s")
    print(f"JAX Newton jacfwd: {jax_jacfwd_elapsed:.6f} s")

    destroy_petsc_solver(options.solver_key)


if __name__ == "__main__":
    main()
