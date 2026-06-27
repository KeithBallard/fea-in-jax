import sys
import time
from pathlib import Path

import numpy as np

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

from testJaxPETScCallJax import petsc_solve_calling_jax



jax.config.update("jax_enable_x64", True)


V4_DIR = Path(__file__).resolve().parents[2] / "v4"
sys.path.insert(0, str(V4_DIR))

import buildKSP_Keith  # noqa: E402
import differentiateKSP_Keith  # noqa: E402


def make_diag(n):
    i = jnp.arange(n, dtype=jnp.float64)
    return 2.0 + jnp.mod(17.0 * i + 3.0, 31.0)


def diagonal_coo(diag):
    n = diag.shape[0]
    shape = jnp.array([n, n], dtype=jnp.int64)
    rows = jnp.arange(n, dtype=jnp.int32)
    cols = jnp.arange(n, dtype=jnp.int32)
    return shape, diag, rows, cols


def make_jac(diag):
    def jac(_x0):
        return diagonal_coo(diag)

    return jac


def v4_no_copy_solve(diag, b):
    shape, vals, rows, cols = diagonal_coo(diag)

    matrix = buildKSP_Keith.linearMatrixInit(jac=make_jac(vals), x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)

    try:
        return differentiateKSP_Keith.linearSolverSolve(
            solver,
            pc,
            shape,
            rows,
            cols,
            vals,
            b,
        )
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


@jax.jit
def jitted_v5_jax_petsc_jax(diag, b):
    return petsc_solve_calling_jax(diag, b)


@jax.jit
def jitted_v4_persistent_no_copy(solver_handle, pc_handle, diag, b):
    shape, vals, rows, cols = diagonal_coo(diag)
    solver = buildKSP_Keith.__CupyCtx(handle=solver_handle)
    pc = buildKSP_Keith.__CupyCtx(handle=pc_handle)
    return differentiateKSP_Keith.linearSolverSolve(
        solver,
        pc,
        shape,
        rows,
        cols,
        vals,
        b,
    )


def timed(label, fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    out.block_until_ready()
    elapsed = time.perf_counter() - start
    print(f"{label} elapsed: {elapsed:.6f} s")
    return out, elapsed


def print_sample(label, x):
    print(f"{label} first 10:", x[:10])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    diag = make_diag(n)
    b = jnp.ones_like(diag)
    expected = b / diag

    print("diagonal size:", n)
    print_sample("diag", diag)
    print_sample("expected", expected)

    matrix = buildKSP_Keith.linearMatrixInit(jac=make_jac(diag), x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)

    try:
        print("Warming up all paths.")
        jitted_v5_jax_petsc_jax(diag, b).block_until_ready()
        v4_no_copy_solve(diag, b).block_until_ready()
        jitted_v4_persistent_no_copy(solver.handle, pc.handle, diag, b).block_until_ready()

        print("\nComparing JAX -> PETSc -> JAX Python Mat path to v4 no-copy PETSc paths.")
        x_v5, t_v5 = timed("v5 JAX-PETSc-JAX", jitted_v5_jax_petsc_jax, diag, b)
        x_v4_lifecycle, t_v4_lifecycle = timed("v4 no-copy lifecycle", v4_no_copy_solve, diag, b)
        x_v4_persistent, t_v4_persistent = timed(
            "v4 no-copy persistent solve",
            jitted_v4_persistent_no_copy,
            solver.handle,
            pc.handle,
            diag,
            b,
        )

        print_sample("v5 x", x_v5)
        print_sample("v4 lifecycle x", x_v4_lifecycle)
        print_sample("v4 persistent x", x_v4_persistent)
        print_sample("expected", expected)
        print("v5-v4 lifecycle norm:", jnp.linalg.norm(x_v5 - x_v4_lifecycle))
        print("v5-v4 persistent norm:", jnp.linalg.norm(x_v5 - x_v4_persistent))
        print("v5/expected norm:", jnp.linalg.norm(x_v5 - expected))
        print("v4 lifecycle/expected norm:", jnp.linalg.norm(x_v4_lifecycle - expected))
        print("v4 persistent/expected norm:", jnp.linalg.norm(x_v4_persistent - expected))
        print("elapsed ratio v5/v4 lifecycle:", t_v5 / t_v4_lifecycle if t_v4_lifecycle else np.inf)
        print("elapsed ratio v5/v4 persistent:", t_v5 / t_v4_persistent if t_v4_persistent else np.inf)

        np.testing.assert_allclose(np.asarray(x_v5), np.asarray(expected), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(x_v4_lifecycle), np.asarray(expected), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(x_v4_persistent), np.asarray(expected), rtol=1e-10, atol=1e-10)
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


if __name__ == "__main__":
    main()
