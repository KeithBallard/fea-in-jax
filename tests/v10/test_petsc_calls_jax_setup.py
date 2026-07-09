"""Smoke test for the v9 PETSc KSP -> JAX matvec direction package.

Run from the PETScJVP root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_petsc_calls_jax_setup.py
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

from v10.JaxCallsPETSc import PETScKSPType, PETScMethodOptions, PETScPCType
from v10.PETScCallsJax import (
    buildSolverObjects,
    cleanupSolverObjects,
    cleanup_jax_mat,
    cleanup_ksp,
    init_jax_mat,
    init_ksp_for_jax_mat,
    petsc_solve,
    solveWithSolverObjects,
)


jax.config.update("jax_enable_x64", True)


def main():
    diag = jnp.array([4.0, 3.0, 9.0, 3.0, 4.0, 8.0, 6.0, 4.0], dtype=jnp.float64)
    b = jnp.ones_like(diag)
    expected = b / diag
    calls = {"manual": 0, "bundled": 0}

    @jax.jit
    def diagonal_matvec(x):
        return diag * x

    def counted_manual_matvec(x):
        calls["manual"] += 1
        return diagonal_matvec(x)

    def counted_bundled_matvec(x):
        calls["bundled"] += 1
        return diagonal_matvec(x)

    options = PETScMethodOptions(
        ksp_type=PETScKSPType.GMRES,
        pc_type=PETScPCType.NONE,
    )

    print("Testing level-3 manual PETSc KSP -> JAX Mat.mult setup.")
    jax_mat = init_jax_mat((diag.shape[0], diag.shape[0]), counted_manual_matvec)
    ksp = init_ksp_for_jax_mat(jax_mat, options)
    ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=100)
    try:
        x_manual = petsc_solve(ksp, b, print_info=True)
        print("manual x:", np.asarray(x_manual))
        print("expected:", np.asarray(expected))
        print("manual JAX matvec calls:", calls["manual"])
        np.testing.assert_allclose(np.asarray(x_manual), np.asarray(expected), rtol=1e-10, atol=1e-10)
        assert calls["manual"] > 0
    finally:
        cleanup_ksp(ksp)
        cleanup_jax_mat(jax_mat)

    print("Testing level-2 bundled JAX -> PETSc KSP -> JAX Mat.mult call.")
    solver_objects = buildSolverObjects(
        (diag.shape[0], diag.shape[0]),
        counted_bundled_matvec,
        options,
    )
    solver_objects.ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=100)
    try:
        x_bundled = solveWithSolverObjects(solver_objects, b, print_info=True)
        x_bundled.block_until_ready()
        print("bundled x:", np.asarray(x_bundled))
        print("expected:", np.asarray(expected))
        print("bundled JAX matvec calls:", calls["bundled"])
        np.testing.assert_allclose(np.asarray(x_bundled), np.asarray(expected), rtol=1e-10, atol=1e-10)
        assert calls["bundled"] > 0
    finally:
        cleanupSolverObjects(solver_objects)


if __name__ == "__main__":
    main()
