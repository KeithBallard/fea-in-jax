"""Smoke test for JAX matrix function -> PETSc Mat callback conversion.

Run from the PETScJVP root with the PETSc/JAX/PETSc environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_jax_mat_func_to_petsc_mat_func.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from petsc4py import PETSc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.NonlinearSNES import (
    convertJAXMatFuncToPETScMatFunc,
    jaxArrayToPETScVec,
)


jax.config.update("jax_enable_x64", True)


def example_residual(x):
    return jnp.array(
        [
            x[0] ** 2 + 3.0 * x[1] - x[2],
            x[0] * x[1] + jnp.sin(x[2]),
            x[2] ** 3 - 2.0 * x[0],
        ],
        dtype=x.dtype,
    )


def main():
    x0 = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float64)
    test_vec = jnp.array([1.0, 2.0, -3.0], dtype=jnp.float64)
    jac_func = jax.jacfwd(example_residual)
    dense_jac = jac_func(x0)
    expected_action = dense_jac @ test_vec

    X = jaxArrayToPETScVec(x0)
    V = jaxArrayToPETScVec(test_vec)
    expected_vec = jaxArrayToPETScVec(expected_action)
    Y = expected_vec.duplicate()
    mat = PETSc.Mat().create(PETSc.COMM_WORLD)

    try:
        print("Testing JAX dense matrix function -> PETSc Mat callback.")
        callback_func = convertJAXMatFuncToPETScMatFunc(jac_func)
        callback_func(None, X, mat, mat, None)
        mat.assemble()
        mat.mult(V, Y)

        diff = Y.duplicate()
        Y.copy(diff)
        diff.axpy(-1.0, expected_vec)
        error_norm = diff.norm()

        print("x0:", x0)
        print("test vector:", test_vec)
        print("dense jacobian:")
        print(dense_jac)
        print("expected J @ v:", expected_action)
        print("PETSc-space action error norm:", error_norm)

        assert error_norm < 1e-12
        diff.destroy()
    finally:
        X.destroy()
        V.destroy()
        Y.destroy()
        expected_vec.destroy()
        mat.destroy()


if __name__ == "__main__":
    main()
