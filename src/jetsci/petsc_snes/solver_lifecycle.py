from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp

from ..options import *
from ..conversions import *
from .solver import *


from petsc4py import PETSc

_PETSC_KSP_TYPES = {
    PETScLinearSolverType.CG: "cg",
    PETScLinearSolverType.LGMRES: "lgmres",
    PETScLinearSolverType.BCGS: "bcgs",
    PETScLinearSolverType.PREONLY: "preonly",
}

_PETSC_PC_TYPES = {
    PETScPreconditionerType.NONE: "none",
    PETScPreconditionerType.JACOBI: "jacobi",
    PETScPreconditionerType.ILU: "ilu",
}


# Stores a map from a key to the solver object, allowing reuse between nonlinear solve calls.
# TODO: convert the tuple value to a small dataclass record once the SNES/IFT
# ownership split settles down.
__solver_dict = {}
__solver_idNum = 0


def _new_solver_key():
    global __solver_idNum
    __solver_idNum += (
        1  # do not base on dict size alone, otherwise you'll get overwriting
    )
    return __solver_idNum


def get_petsc_solver_objects_from_key(solver_key: int):
    """Return the live SNES and companion KSP wrappers for `solver_key`."""
    if solver_key not in __solver_dict:
        raise KeyError(f"No PETSc solver found for solver_key={solver_key}")
    return __solver_dict[solver_key]


def _coo_jacobian_function(R: Callable, J: Callable | None):
    """Return a function of x that produces COOData for the SNES Jacobian."""
    if J is None:

        def jacobian_coo_from_residual(x):
            J = jax.jacfwd(R)(x)
            print(J)
            return convert_jax_dense_mat_to_coo_data(J)

        return jacobian_coo_from_residual

    def jacobian_coo(x):
        jacobian = J(x)
        return convert_jax_mat_to_coo_data(jacobian)

    return jacobian_coo


def _apply_snes_options(snes, options: SolverOptions):
    snes.setTolerances(
        rtol=options.nonlinear_relative_tol,
        atol=options.nonlinear_absolute_tol,
        stol=options.nonlinear_step_tol,
        max_it=options.nonlinear_max_iter,
    )


def _apply_ksp_options(snes, options: SolverOptions):
    ksp = snes.getKSP()
    ksp.setType(_PETSC_KSP_TYPES[options.linear_solve_type])
    if hasattr(PETSc.KSP, "NormType"):
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=options.linear_relative_tol,
        atol=options.linear_absolute_tol,
        max_it=options.linear_max_iter,
    )
    pc = ksp.getPC()
    pc.setType(_PETSC_PC_TYPES[options.linear_precond_type])


def _apply_ksp_options_direct(ksp, options: SolverOptions):
    """Apply KSP/PC options to a standalone PETSc KSP object."""
    ksp.setType(_PETSC_KSP_TYPES[options.linear_solve_type])
    if hasattr(PETSc.KSP, "NormType"):
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=options.linear_relative_tol,
        atol=options.linear_absolute_tol,
        max_it=options.linear_max_iter,
    )
    pc = ksp.getPC()
    pc.setType(_PETSC_PC_TYPES[options.linear_precond_type])


def build_petsc_snes_from_options(
    R: Callable, J: Callable | None, options: SolverOptions
):
    """Build a PETSc SNES solver from JAX residual/Jacobian functions.

    `R` is expected to be a JAX function of the nonlinear state `x`. If `J` is
    provided it may return either COOData or a dense rank-2 JAX matrix. If `J`
    is `None`, a dense Jacobian is built with `jax.jacfwd(R)` for now.
    """
    if options.nonlinear_solver_type is not NonlinearSolverType.PETSC_SNES:
        raise TypeError("build_petsc_snes_from_options only builds PETSc SNES solvers")

    callback_stats = {}
    residual_callback = convert_jax_vec_func_to_petsc_vec_func(
        R,
        stats=callback_stats,
    )
    jacobian_callback_state = PatternAwareMatAssignmentState()
    jacobian_callback = convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
        _coo_jacobian_function(R, J),
        state=jacobian_callback_state,
        stats=callback_stats,
    )

    snes = PETSc.SNES().create(PETSc.COMM_WORLD)
    _apply_snes_options(snes, options)
    _apply_ksp_options(snes, options)

    return PETScNonlinearSolver(
        snes=snes,
        residual_callback=residual_callback,
        jacobian_callback=jacobian_callback,
        options=options,
        jacobian_callback_state=jacobian_callback_state,
        callback_stats=callback_stats,
    )


def build_petsc_internal_ksp_from_options(options: SolverOptions):
    """Build a standalone PETSc KSP wrapper for IFT-style linear solves.

    This companion object is intentionally separate from the SNES-owned KSP so
    the differentiation path can reuse PETSc linear algebra without borrowing
    state from the primal nonlinear solve.
    """
    if options.nonlinear_solver_type is not NonlinearSolverType.PETSC_SNES:
        raise TypeError("build_petsc_internal_ksp_from_options only builds PETSc KSP solvers")

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    _apply_ksp_options_direct(ksp, options)

    return PETScLinearSolver(
        ksp=ksp,
        vector_callback=lambda *args, **kwargs: None,
        matrix_callback=lambda *args, **kwargs: None,
        options=options,
    )


def build_petsc_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J: jax.tree_util.Partial,
):
    """Return a solver and SolverOptions containing its dictionary key.

    If `options.solver_key` is `None`, a new PETSc solver is built and stored.
    If a key is present, the existing solver is retrieved and refreshed with
    the latest callbacks and method options.
    """

    validate_petsc_solver_options(options)

    if options.solver_key is None:
        solver = build_petsc_snes_from_options(R, J, options)
        ksp_for_IFT = build_petsc_internal_ksp_from_options(options)
        solver_key = _new_solver_key()
        __solver_dict[solver_key] = (
            solver,
            ksp_for_IFT,
        )  # this way we hide the KSP since we only need it for the KSP
        return solver, replace(options, solver_key=solver_key)
    else:
        solver, ksp_for_IFT = get_petsc_solver_objects_from_key(options.solver_key)
        update_petsc_snes_callbacks(solver, R, J)
        update_petsc_snes_options(solver, options) #These may need to be reworked but for now lets just assume on reuse you want to change the funcitons
        update_petsc_linear_solver_options(ksp_for_IFT, options)

    return solver, options


def update_petsc_snes_callbacks(
    solver: PETScNonlinearSolver,
    R: Callable,
    J: Callable | None,
):
    """Replace residual/Jacobian callbacks on an existing PETSc solver."""
    if solver.callback_stats is None:
        solver.callback_stats = {}
    solver.residual_callback = convert_jax_vec_func_to_petsc_vec_func(
        R,
        stats=solver.callback_stats,
    )
    if solver.jacobian_callback_state is None:
        solver.jacobian_callback_state = PatternAwareMatAssignmentState()
    solver.jacobian_callback = convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
        _coo_jacobian_function(R, J),
        state=solver.jacobian_callback_state,
        stats=solver.callback_stats,
    )
    if solver.residual_vec is not None:
        solver.snes.setFunction(solver.residual_callback, solver.residual_vec)
    if solver.jacobian_mat is not None:
        solver.snes.setJacobian(
            solver.jacobian_callback,
            solver.jacobian_mat,
            solver.jacobian_mat,
        )
    return solver


def update_petsc_snes_options(solver: PETScNonlinearSolver, options: SolverOptions):
    """Apply new PETSc method/tolerance options to an existing solver."""
    solver.options = options
    _apply_snes_options(solver.snes, options)
    _apply_ksp_options(solver.snes, options)
    return solver


def update_petsc_linear_solver_options(
    solver: PETScLinearSolver,
    options: SolverOptions,
):
    """Apply new PETSc method/tolerance options to an existing KSP wrapper."""
    solver.options = options
    _apply_ksp_options_direct(solver.ksp, options)
    return solver


def destroy_petsc_solver(solver_key: int):
    """Remove a solver from the dictionary and destroy its PETSc objects."""
    solver = __solver_dict.pop(solver_key, None)
    if solver is None:
        return None
    solver[0].destroy()
    solver[1].destroy()
    if hasattr(PETSc, "garbage_cleanup"):
        PETSc.garbage_cleanup()
    return solver

    # careful with this, because it can let you overwriting existing solvers in it's current state.
    # If you have 2 solvers and pop number 1 the next id will be 2 which will overwrite
    # it may be better to move to an increasing number system
