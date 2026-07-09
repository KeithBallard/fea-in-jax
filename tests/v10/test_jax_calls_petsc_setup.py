"""Smoke test for the v9 JAX -> PETSc KSP direction package.

Run from the PETScJVP root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_jax_calls_petsc_setup.py
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
    cleanup_ksp,
    cleanup_matrix,
    cleanup_pc,
    init_ksp,
    init_matrix_from_function,
    init_pc,
    solveWithSolverObjects,
    solve_ksp,
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


def main():
    diag = jnp.array([4.0, 3.0, 9.0, 3.0, 4.0, 8.0, 6.0, 4.0], dtype=jnp.float64)
    b = jnp.ones_like(diag)
    x0 = jnp.zeros_like(diag)
    expected = b / diag
    matrix_data = diagonal_coo_data(diag)
    options = PETScMethodOptions(
        ksp_type=PETScKSPType.CG,
        pc_type=PETScPCType.JACOBI,
    )

    print("Testing level-2 JaxCallsPETSc object bundle.")
    solver_objects = buildSolverObjects(diagonal_matrix_function(diag), x0, options)
    try:
        x_level2 = solveWithSolverObjects(solver_objects, b)
        x_level2.block_until_ready()
        print("level-2 x:", np.asarray(x_level2))
        print("expected:", np.asarray(expected))
        np.testing.assert_allclose(np.asarray(x_level2), np.asarray(expected), rtol=1e-10, atol=1e-10)
    finally:
        cleanupSolverObjects(solver_objects)

    print("Testing level-3 manual Mat/PC/KSP setup.")
    matrix = init_matrix_from_function(diagonal_matrix_function(diag), x0, options)
    pc = init_pc(matrix, options)
    ksp = init_ksp(matrix, options)
    try:
        x_level3 = solve_ksp(ksp, pc, matrix_data, b)
        x_level3.block_until_ready()
        print("level-3 x:", np.asarray(x_level3))
        print("expected:", np.asarray(expected))
        np.testing.assert_allclose(np.asarray(x_level3), np.asarray(expected), rtol=1e-10, atol=1e-10)
    finally:
        cleanup_ksp(ksp)
        cleanup_pc(pc)
        cleanup_matrix(matrix)


if __name__ == "__main__":
    main()
