"""Smoke test for direct-only pattern-aware PETSc Mat callbacks.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_snes_direct_only_mat_pattern_aware.py
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

from v10.JaxCallsPETSc.linear_methods import COOData
from v10.NonlinearSNES.direct_mat_function_converters import (
    convertJAXCOOFuncToPETScMatFuncDirectPatternAware,
)
from v10.NonlinearSNES.direct_vec_function_converters import jaxArrayToPETScVec


jax.config.update("jax_enable_x64", True)


def pattern_switching_coo_func(x):
    n = x.shape[0]
    idx = jnp.arange(n, dtype=jnp.int32)
    use_tridiagonal = bool((x[0] > 5.0).block_until_ready())

    diag_rows = idx
    diag_cols = idx
    diag_vals = 2.0 + x

    if use_tridiagonal:
        off_rows = jnp.arange(n - 1, dtype=jnp.int32)
        off_cols = off_rows + 1
        off_vals = 0.1 * jnp.ones((n - 1,), dtype=x.dtype)
        rows = jnp.concatenate([diag_rows, off_rows])
        cols = jnp.concatenate([diag_cols, off_cols])
        vals = jnp.concatenate([diag_vals, off_vals])
    else:
        rows = diag_rows
        cols = diag_cols
        vals = diag_vals

    return COOData(
        shape=jnp.asarray((n, n), dtype=jnp.int64),
        vals=vals,
        rows=rows,
        cols=cols,
    )


def expected_action(x, test_vec):
    result = (2.0 + x) * test_vec
    if bool((x[0] > 5.0).block_until_ready()):
        return result.at[:-1].add(0.1 * test_vec[1:])
    return result


def check_action(mat, x, test_vec, *, label):
    expected = expected_action(x, test_vec)
    V = jaxArrayToPETScVec(test_vec)
    expected_vec = jaxArrayToPETScVec(expected)
    Y = expected_vec.duplicate()
    diff = None
    try:
        mat.mult(V, Y)
        diff = Y.duplicate()
        Y.copy(diff)
        diff.axpy(-1.0, expected_vec)
        error_norm = diff.norm()
        print(f"{label} expected:", expected)
        print(f"{label} PETSc-space action error norm:", error_norm)
        assert error_norm < 1e-12
    finally:
        if diff is not None:
            diff.destroy()
        V.destroy()
        expected_vec.destroy()
        Y.destroy()


def main():
    x_same_pattern_0 = jnp.array([1.0, 4.0, 2.0, 8.0], dtype=jnp.float64)
    x_same_pattern_1 = jnp.array([3.0, 2.0, 6.0, 1.0], dtype=jnp.float64)
    x_changed_pattern = jnp.array([6.0, 2.0, 1.0, 4.0], dtype=jnp.float64)
    test_vec = jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    X0 = jaxArrayToPETScVec(x_same_pattern_0)
    X1 = jaxArrayToPETScVec(x_same_pattern_1)
    X2 = jaxArrayToPETScVec(x_changed_pattern)
    mat = PETSc.Mat().create(PETSc.COMM_WORLD)

    try:
        print("Testing direct-only pattern-aware COOData -> PETSc Mat callback.")
        callback_func = convertJAXCOOFuncToPETScMatFuncDirectPatternAware(pattern_switching_coo_func)

        callback_func(None, X0, mat, mat, None)
        mat.assemble()
        check_action(mat, x_same_pattern_0, test_vec, label="first diagonal pattern")

        callback_func(None, X1, mat, mat, None)
        mat.assemble()
        check_action(mat, x_same_pattern_1, test_vec, label="second diagonal values-only")

        callback_func(None, X2, mat, mat, None)
        mat.assemble()
        check_action(mat, x_changed_pattern, test_vec, label="third tridiagonal rebuild")
    finally:
        X0.destroy()
        X1.destroy()
        X2.destroy()
        mat.destroy()


if __name__ == "__main__":
    main()
