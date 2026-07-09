"""Level-2 workflows for PETSc KSPs that call JAX matvecs."""

from __future__ import annotations

from dataclasses import dataclass

from ..JaxCallsPETSc.options import PETScMethodOptions
from .linear_methods import (
    DEFAULT_KSP_CALL_JAX_OPTIONS,
    JaxMatObjects,
    cleanup_jax_mat,
    cleanup_ksp,
    init_jax_mat,
    init_ksp_for_jax_mat,
    jax_call_petsc_solve,
    petsc_solve,
)


#This is Jax calling PETSc which calls Jax again. Don't do this


@dataclass(frozen=True)
class KSPCallingJaxObjects:
    """Live objects for the PETSc -> JAX matvec path."""

    jax_mat: JaxMatObjects
    ksp: object
    options: PETScMethodOptions


def buildSolverObjects(shape, matvec, options: PETScMethodOptions | None = None, *, dmplex=None, comm=None):
    """Build a JAX-backed PETSc Python Mat and KSP."""
    options = DEFAULT_KSP_CALL_JAX_OPTIONS if options is None else options
    jax_mat = init_jax_mat(shape, matvec, dmplex=dmplex, comm=comm)
    ksp = init_ksp_for_jax_mat(jax_mat, options, comm=comm)
    return KSPCallingJaxObjects(jax_mat=jax_mat, ksp=ksp, options=options)


def solveWithSolverObjects(solver_objects: KSPCallingJaxObjects, b, *, call_from_jax=True, print_info=False):
    """Solve with a PETSc KSP whose Mat.mult calls JAX."""
    if call_from_jax:
        return jax_call_petsc_solve(
            solver_objects.ksp,
            b,
            print_info=print_info,
        )

    return petsc_solve(
        solver_objects.ksp,
        b,
        print_info=print_info,
    )


def runSimulationWithSolverObjects(
    solver_objects: KSPCallingJaxObjects,
    rhs_sequence,
    *,
    call_from_jax=True,
    print_info=False,
):
    """Run a sequence of PETSc solves using one JAX-backed KSP."""
    return [
        solveWithSolverObjects(
            solver_objects,
            rhs,
            call_from_jax=call_from_jax,
            print_info=print_info,
        )
        for rhs in rhs_sequence
    ]


def cleanupSolverObjects(solver_objects: KSPCallingJaxObjects):
    """Destroy the KSP and JAX-backed PETSc Python Mat."""
    cleanup_ksp(solver_objects.ksp)
    cleanup_jax_mat(solver_objects.jax_mat)
    return solver_objects


build_solver_objects = buildSolverObjects
solve_with_solver_objects = solveWithSolverObjects
run_simulation_with_solver_objects = runSimulationWithSolverObjects
cleanup_solver_objects = cleanupSolverObjects
