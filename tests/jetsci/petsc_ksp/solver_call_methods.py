"""Level-2 linear solver orchestration.

This layer batches level-3 PETSc method calls into the workflows we expect
nonlinear/simulation code to use most often:

- build the PETSc Mat/PC/KSP object bundle once
- reuse that bundle for one or more linear solves
- optionally update matrix values when the sparsity pattern stays fixed
- cleanup the bundle in one call
"""

from __future__ import annotations

from dataclasses import dataclass

from .linear_methods import (
    COOData,
    cleanup_ksp,
    cleanup_matrix,
    cleanup_pc,
    evaluate_matrix_function,
    init_ksp,
    init_matrix_from_COOData,
    init_pc,
    solve_ksp_from_coo_data,
    to_COOData_object,
    update_matrix_values,
)
from .options import PETScMethodOptions


@dataclass(frozen=True)
class PETScLinearSolverObjects:
    """Live PETSc objects plus the JAX-visible matrix data they represent."""

    matrix: object
    pc: object
    ksp: object
    matrix_data: COOData
    options: PETScMethodOptions


def buildSolverObjects(matrix_function, x, options: PETScMethodOptions = PETScMethodOptions()):
    """Build Mat/PC/KSP objects from a matrix-producing callable.

    `matrix_function(x)` may return either a `COOData` object or a JAX sparse
    COO matrix. The returned `PETScLinearSolverObjects` owns live PETSc objects
    and should be cleaned up with `cleanupSolverObjects`.
    """
    matrix_data = evaluate_matrix_function(matrix_function, x)
    matrix = init_matrix_from_COOData(matrix_data, options)
    pc = init_pc(matrix, options)
    ksp = init_ksp(matrix, options)
    return PETScLinearSolverObjects(
        matrix=matrix,
        pc=pc,
        ksp=ksp,
        matrix_data=matrix_data,
        options=options,
    )


def buildSolverObjectsFromCOOData(data: COOData, options: PETScMethodOptions = PETScMethodOptions()):
    """Build Mat/PC/KSP objects from already-evaluated COO data."""
    matrix_data = to_COOData_object(data)
    matrix = init_matrix_from_COOData(matrix_data, options)
    pc = init_pc(matrix, options)
    ksp = init_ksp(matrix, options)
    return PETScLinearSolverObjects(
        matrix=matrix,
        pc=pc,
        ksp=ksp,
        matrix_data=matrix_data,
        options=options,
    )


def solveWithSolverObjects(
    solver_objects: PETScLinearSolverObjects,
    b,
    matrix_data=None,
    *,
    matrix_function=None,
    x=None,
    update_matrix: bool = False,
):
    """Solve one linear system using an existing PETSc object bundle.

    If `matrix_data` or `matrix_function` is supplied, those values become the
    JAX-visible matrix for the solve. Set `update_matrix=True` when the PETSc
    Mat should also be updated in-place with those values.
    """
    if matrix_function is not None:
        visible_matrix_data = evaluate_matrix_function(matrix_function, x)
    elif matrix_data is not None:
        visible_matrix_data = to_COOData_object(matrix_data)
    else:
        visible_matrix_data = solver_objects.matrix_data

    if update_matrix:
        update_matrix_values(solver_objects.matrix, visible_matrix_data.vals)

    return solve_ksp_from_coo_data(
        solver_objects.ksp,
        solver_objects.pc,
        visible_matrix_data,
        b,
    )


def runSimulationWithSolverObjects(
    solver_objects: PETScLinearSolverObjects,
    rhs_sequence,
    *,
    matrix_data_sequence=None,
    update_matrix: bool = False,
):
    """Run a simple sequence of linear solves with one PETSc object bundle.

    This intentionally stays minimal: higher nonlinear code can decide what a
    timestep or Newton iteration means, while this helper only batches repeated
    calls to `solveWithSolverObjects`.
    """
    if matrix_data_sequence is None:
        return [
            solveWithSolverObjects(solver_objects, rhs)
            for rhs in rhs_sequence
        ]

    return [
        solveWithSolverObjects(
            solver_objects,
            rhs,
            matrix_data=matrix_data,
            update_matrix=update_matrix,
        )
        for rhs, matrix_data in zip(rhs_sequence, matrix_data_sequence)
    ]


def cleanupSolverObjects(solver_objects: PETScLinearSolverObjects):
    """Destroy the PETSc objects owned by a solver bundle."""
    cleanup_ksp(solver_objects.ksp)
    cleanup_pc(solver_objects.pc)
    cleanup_matrix(solver_objects.matrix)
    return solver_objects


build_solver_objects = buildSolverObjects
build_solver_objects_from_coo_data = buildSolverObjectsFromCOOData
solve_with_solver_objects = solveWithSolverObjects
run_simulation_with_solver_objects = runSimulationWithSolverObjects
cleanup_solver_objects = cleanupSolverObjects
