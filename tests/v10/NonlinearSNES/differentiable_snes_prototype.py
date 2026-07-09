"""Prototype contract for differentiating PETSc SNES solves with IFT.

This file is intentionally not the final SNES implementation. It captures the
shape we want:

1. The primal solve exposes `phi`/`args` as a JAX-visible argument.
2. PETSc still sees residual/Jacobian callbacks that are functions of `x`.
3. Differentiation uses the implicit function theorem at the converged state.

Mathematically, if

    R(x_star, phi) = 0

then the JVP with respect to `phi` solves

    R_x x_dot = -R_phi phi_dot

where both derivatives are evaluated at `(x_star, phi)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .direct_mat_function_converters import (
    convertJAXCOOFuncToPETScMatFuncDirectPatternAware,
    convertJaxMatToCOOData,
)
from .direct_vec_function_converters import convertJAXVecFuncToPETScVecFuncDirect
from .direct_vec_function_converters import jaxArrayToPETScVec, petscVecToJAX


Array = Any
ResidualFunction = Callable[[Array, Any], Array]


@dataclass(frozen=True)
class DifferentiableSNESHooks:
    """Runtime hooks needed by the differentiable SNES wrapper.

    These hooks are deliberately external to the autodiff rule so v10 can
    evolve the PETSc object lifecycle independently from the IFT contract.
    """

    nonlinear_solve: Callable[..., Array]
    linear_solve: Callable[..., Array]


def dense_jacobian_coo_func(residual_func: ResidualFunction, phi):
    """Build a COO-producing `dR/dx` function for a fixed `phi`.

    This is a simple prototype helper. It uses `jax.jacfwd` and dense-to-COO
    conversion, which is good for smoke tests but not the sparse production
    path we ultimately want.
    """

    def jacobian_coo(x):
        jac_x_dense = jax.jacfwd(lambda state: residual_func(state, phi))(x)
        return convertJaxMatToCOOData(jac_x_dense)

    return jacobian_coo


def make_petsc_snes_callbacks(residual_func: ResidualFunction, phi):
    """Create PETSc residual/Jacobian callbacks with `phi` closed over.

    PETSc sees functions of `x`; JAX still sees `phi` at the wrapper level.
    """

    def residual_x(x):
        return residual_func(x, phi)

    jacobian_coo_x = dense_jacobian_coo_func(residual_func, phi)

    petsc_residual = convertJAXVecFuncToPETScVecFuncDirect(residual_x)
    petsc_jacobian = convertJAXCOOFuncToPETScMatFuncDirectPatternAware(jacobian_coo_x)
    return residual_x, jacobian_coo_x, petsc_residual, petsc_jacobian


def differentiablePETScSolvePrototype(
    residual_func: ResidualFunction,
    hooks: DifferentiableSNESHooks,
    *,
    options=None,
):
    """Create a differentiable solve function `solve(phi, x0)`.

    Expected hook signatures for this prototype:

    ```python
    nonlinear_solve(
        residual_x,
        jacobian_coo_x,
        petsc_residual,
        petsc_jacobian,
        x0,
        options,
    ) -> x_star

    linear_solve(
        jacobian_coo_func,
        x_star,
        phi,
        rhs,
        options,
    ) -> solution
    ```

    `nonlinear_solve` receives both raw JAX callables and PETSc callbacks. A
    pure-JAX test hook can use the raw callables; a real PETSc SNES hook should
    use the PETSc callbacks. `linear_solve` is where we will eventually reuse
    PETSc KSP on the final `R_x` matrix. For reverse mode through this custom
    JVP, that linear solve must ultimately be a primitive with a transpose rule.
    """

    @jax.custom_jvp
    def solve(phi, x0):
        residual_x, jacobian_coo_x, petsc_residual, petsc_jacobian = make_petsc_snes_callbacks(
            residual_func,
            phi,
        )
        x_star = hooks.nonlinear_solve(
            residual_x,
            jacobian_coo_x,
            petsc_residual,
            petsc_jacobian,
            jax.lax.stop_gradient(x0),
            options,
        )
        return jax.lax.stop_gradient(x_star)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        phi, x0 = primals
        phi_dot, _x0_dot = tangents

        x_star = solve(phi, x0)

        def residual_at_solution(active_phi):
            return residual_func(x_star, active_phi)

        _, residual_phi_dot = jax.jvp(
            residual_at_solution,
            (phi,),
            (phi_dot,),
        )
        rhs = -residual_phi_dot

        jacobian_coo_x = dense_jacobian_coo_func(residual_func, phi)
        x_dot = hooks.linear_solve(
            jacobian_coo_x,
            x_star,
            phi,
            rhs,
            options,
        )

        return x_star, x_dot

    return solve


def pure_jax_newton_solve_for_testing(
    residual_x,
    jacobian_x,
    petsc_residual,
    petsc_jacobian,
    x0,
    options=None,
):
    """Tiny Newton solver hook for local/non-PETSc prototype tests."""
    del petsc_residual, petsc_jacobian, options
    x = x0
    for _ in range(8):
        r = residual_x(x)
        jac_data = jacobian_x(x)
        jac_dense = jnp.zeros(tuple(jac_data.shape), dtype=jac_data.vals.dtype)
        jac_dense = jac_dense.at[jac_data.rows, jac_data.cols].set(jac_data.vals)
        dx = jnp.linalg.solve(jac_dense, -r)
        x = x + dx
    return x


def pure_jax_linear_solve_for_testing(jacobian_coo_x, x_star, phi, rhs, options=None):
    """Tiny dense linear solve hook matching the prototype hook signature."""
    del phi, options
    jac_data = jacobian_coo_x(x_star)
    jac_dense = jnp.zeros(tuple(jac_data.shape), dtype=jac_data.vals.dtype)
    jac_dense = jac_dense.at[jac_data.rows, jac_data.cols].set(jac_data.vals)
    return jnp.linalg.solve(jac_dense, rhs)


def petsc_snes_solve_for_prototype(
    residual_x,
    jacobian_x,
    petsc_residual,
    petsc_jacobian,
    x0,
    options=None,
):
    """Minimal PETSc SNES nonlinear solve hook for the prototype.

    This hook uses the PETSc callbacks generated by `make_petsc_snes_callbacks`
    and returns a JAX array containing the converged solution. The PETSc solve
    mutates a PETSc-owned duplicate of the initial vector, not the original JAX
    `x0` buffer.
    """
    del residual_x, jacobian_x
    from petsc4py import PETSc

    options = options or {}
    x0_view = jaxArrayToPETScVec(x0)
    x = x0_view.duplicate()
    r = x0_view.duplicate()
    jac = PETSc.Mat().create(PETSc.COMM_WORLD)
    snes = PETSc.SNES().create(PETSc.COMM_WORLD)
    try:
        x0_view.copy(x)
        snes.setFunction(petsc_residual, r)
        snes.setJacobian(petsc_jacobian, jac, jac)

        if "type" in options:
            snes.setType(options["type"])
        if "rtol" in options or "atol" in options or "stol" in options or "max_it" in options:
            snes.setTolerances(
                rtol=options.get("rtol"),
                atol=options.get("atol"),
                stol=options.get("stol"),
                max_it=options.get("max_it"),
            )
        if options.get("set_from_options", False):
            snes.setFromOptions()

        snes.solve(None, x)
        x_out = petscVecToJAX(x).copy()
        x_out.block_until_ready()
        return x_out
    finally:
        snes.destroy()
        jac.destroy()
        r.destroy()
        x.destroy()
        x0_view.destroy()


__all__ = [
    "DifferentiableSNESHooks",
    "dense_jacobian_coo_func",
    "differentiablePETScSolvePrototype",
    "make_petsc_snes_callbacks",
    "petsc_snes_solve_for_prototype",
    "pure_jax_linear_solve_for_testing",
    "pure_jax_newton_solve_for_testing",
]
