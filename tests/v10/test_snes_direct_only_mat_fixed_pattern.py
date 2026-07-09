"""Smoke test for direct-only fixed-pattern PETSc Mat callbacks.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_snes_direct_only_mat_fixed_pattern.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from petsc4py import PETSc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.JaxCallsPETSc.linear_methods import COOData
from v10.NonlinearSNES.direct_mat_function_converters import (
    convertJAXCOOFuncToPETScMatFuncDirectFixedPattern,
)
from v10.NonlinearSNES.direct_vec_function_converters import jaxArrayToPETScVec


jax.config.update("jax_enable_x64", True)


def diagonal_coo_func(x):
    n = x.shape[0]
    idx = jnp.arange(n, dtype=jnp.int32)
    vals = 2.0 + x
    return COOData(
        shape=jnp.asarray((n, n), dtype=jnp.int64),
        vals=vals,
        rows=idx,
        cols=idx,
    )


def check_action(mat, x, test_vec, *, label):
    expected_action = (2.0 + x) * test_vec
    V = jaxArrayToPETScVec(test_vec)
    expected_vec = jaxArrayToPETScVec(expected_action)
    Y = expected_vec.duplicate()
    diff = None
    try:
        mat.mult(V, Y)
        diff = Y.duplicate()
        Y.copy(diff)
        diff.axpy(-1.0, expected_vec)
        error_norm = diff.norm()
        print(f"{label} expected first 6:", expected_action[:6])
        print(f"{label} PETSc-space action error norm:", error_norm)
        assert error_norm < 1e-12
    finally:
        if diff is not None:
            diff.destroy()
        V.destroy()
        expected_vec.destroy()
        Y.destroy()


def main():
    x0 = jnp.array([1.0, 4.0, 2.0, 8.0, 3.0, 5.0], dtype=jnp.float64)
    x1 = jnp.array([3.0, 2.0, 6.0, 1.0, 7.0, 4.0], dtype=jnp.float64)
    test_vec = jnp.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0], dtype=jnp.float64)

    X0 = jaxArrayToPETScVec(x0)
    X1 = jaxArrayToPETScVec(x1)
    mat = PETSc.Mat().create(PETSc.COMM_WORLD)

    try:
        print("Testing direct-only fixed-pattern COOData -> PETSc Mat callback.")
        callback_func = convertJAXCOOFuncToPETScMatFuncDirectFixedPattern(diagonal_coo_func)

        t0 = time.perf_counter()
        callback_func(None, X0, mat, mat, None)
        mat.assemble()
        first_elapsed = time.perf_counter() - t0
        check_action(mat, x0, test_vec, label="first/preallocation call")

        t1 = time.perf_counter()
        callback_func(None, X1, mat, mat, None)
        mat.assemble()
        second_elapsed = time.perf_counter() - t1
        check_action(mat, x1, test_vec, label="second/values-only call")

        print("first/preallocation elapsed:", first_elapsed)
        print("second/values-only elapsed:", second_elapsed)
        print("same PETSc Mat handle after update:", int(mat.handle))
    finally:
        X0.destroy()
        X1.destroy()
        mat.destroy()


if __name__ == "__main__":
    main()
