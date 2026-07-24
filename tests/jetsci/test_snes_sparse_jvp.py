"""Small sparse PETSc/JAX implicit-differentiation smoke test."""


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

from test_snes_solver_timing import make_system

def main(size: int = 1000):
    _, operator, residual, jacobian, jacobian_coo = make_system(size)
    index = jnp.arange(size, dtype=jnp.float64)
    phi = 4.0 + 0.6 * index + 0.2 * jnp.sin(index + 0.25)
    x0 = 0.1 * jnp.cos(index)
    phi_dot = 0.1 * jnp.sin(index + 0.75)

    def _measure(label: str, thunk, block: bool = False):
        start = perf_counter()
        value = thunk(phi)
        if block and hasattr(value, "block_until_ready"):
            value.block_until_ready()
        elapsed = perf_counter() - start
        print(f"{label}: {elapsed:.6f} s")
        return value, elapsed

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_precond_type=PETScPreconditionerType.JACOBI,
        linear_solve_type=PETScLinearSolverType.LGMRES,
        linear_absolute_tol=1e-14,
    )

    residual_for_phi = jax.tree_util.Partial(residual, phi)
    jacobian_for_phi = jax.tree_util.Partial(jacobian_coo, phi)
    _, options = build_petsc_solver_with_reuse(
        options,
        residual_for_phi,
        jacobian_for_phi,
    )

    primitive = DifferentiableSNESPrimitive(
        residual=residual,
        jacobian=jacobian_coo,
        solver_key=options.solver_key,
    )
    set_jvp_diagnostics(True)
    solve = make_differentiable_snes_solve(primitive)

    jax.block_until_ready(solve(phi, x0)) #warmup run

    print("First solve----------------------")
    x_star_first_measure, pet_primal_elapsed_2 = _measure(
            "PETSc primal solve",
            jax.tree_util.Partial(solve,x0=x0),  #assumes phi
            block=True,
        )



    print("star JVP----------------------")
    x_star, x_dot = jax.jvp(
        jax.tree_util.Partial(solve,x0=x0),
        (phi,),
        (phi_dot,),
    )
    x_star.block_until_ready()
    x_dot.block_until_ready()

    print("residual JVP----------------------")
    residual_phi_dot = jax.jvp(
        jax.tree_util.Partial(residual,x=x_star),
        (phi,),
        (phi_dot,),
    )[1]
    expected_x_dot = jnp.linalg.solve(
        jacobian(phi, x_star),
        -residual_phi_dot,
    )

    print("JacFwd----------------------")
    jacobian_evaluated = jax.jacfwd(jax.tree_util.Partial(solve,x0=x_star))(phi)
    jacobian_evaluated.block_until_ready()
    #print("Jacobian at phi",jacobian_evaluated.transpose()) #we may need to use a transpose solve here instead

    """expected_jac = jax.jacfwd(
            lambda active_phi: jnp.linalg.solve(
                operator(active_phi),
                jnp.sin(active_phi),
            )
    )(phi)
    """

    #print(expected_jac)
    #print("difference between the two evalations",expected_jac - jacobian_evaluated.transpose())
    

    print("JacFwd 2----------------------")
    jacobian_reevaluated = jax.jacfwd(jax.tree_util.Partial(solve,x0=x_star))(phi*2)
    jacobian_reevaluated.block_until_ready()
    #print("Jacobian at second phi",jacobian_reevaluated.transpose())

    """expected_jac_2 = jax.jacfwd(
                lambda active_phi: jnp.linalg.solve(
                    operator(active_phi),
                    jnp.sin(active_phi),
                )
    )(phi*2)"""

    #print(expected_jac_2)
    #print("difference between the two reevalations",expected_jac_2 - jacobian_reevaluated.transpose())
        

    exit(1)

    print("sparse PETSc JVP test")
    print("size:", size)
    print("primal residual norm:", jnp.linalg.norm(residual(phi, x_star)))
    print("JVP error norm:", jnp.linalg.norm(x_dot - expected_x_dot))
    #print(jacobian_evaluated)
    #print(jacobian_reevaluated)

    np.testing.assert_allclose(
        np.asarray(x_dot),
        np.asarray(expected_x_dot),
        rtol=1e-8,
        atol=1e-8,
    )



if __name__ == "__main__":
    main()
