import os
import sys
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


KSP_JAX_DIR = Path(__file__).resolve().parents[1] / "KSP-Jax-rules"
sys.path.insert(0, str(KSP_JAX_DIR))

import buildKSP_Keith  # noqa: E402
import differentiateKSP_Keith  # noqa: E402


def make_diag(n, step):
    i = jnp.arange(n, dtype=jnp.float64)
    base = 2.0 + jnp.mod(17.0 * i + 3.0, 31.0)
    return base + 0.25 * step


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


def rebuild_and_solve(diag, b):
    shape, vals, rows, cols = diagonal_coo(diag)
    matrix = buildKSP_Keith.linearMatrixInit(jac=make_jac(vals), x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)

    try:
        x = differentiateKSP_Keith.linearSolverSolve(
            solver,
            pc,
            shape,
            rows,
            cols,
            vals,
            b,
        )
        x.block_until_ready()
        return x
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


def update_and_solve(matrix_handle, solver_handle, pc_handle, diag, b):
    shape, vals, rows, cols = diagonal_coo(diag)
    matrix = buildKSP_Keith.__CupyCtx(handle=matrix_handle)
    solver = buildKSP_Keith.__CupyCtx(handle=solver_handle)
    pc = buildKSP_Keith.__CupyCtx(handle=pc_handle)

    updated_matrix = buildKSP_Keith.linearMatrixUpdate(matrix, vals)
    updated_matrix.handle.block_until_ready()
    return differentiateKSP_Keith.linearSolverSolve(
        solver,
        pc,
        shape,
        rows,
        cols,
        vals,
        b,
    )


def timed_step(label, fn, *args):
    start = time.perf_counter()
    x = fn(*args)
    x.block_until_ready()
    elapsed = time.perf_counter() - start
    print(f"{label} elapsed: {elapsed:.6f} s")
    return x, elapsed


def print_sample(label, x):
    print(f"{label} first 10:", x[:10])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    b = jnp.ones((n,), dtype=jnp.float64)
    initial_diag = make_diag(n, 0)

    print("diagonal size:", n)
    print("value-change steps:", n_steps)
    print_sample("initial diag", initial_diag)

    matrix = buildKSP_Keith.linearMatrixInit(jac=make_jac(initial_diag), x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)

    rebuild_times = []
    update_times = []

    try:
        print("\nWarming up both paths.")
        rebuild_and_solve(initial_diag, b).block_until_ready()
        update_and_solve(matrix.handle, solver.handle, pc.handle, initial_diag, b).block_until_ready()

        print("\nChanging matrix values while keeping diagonal sparsity fixed.")
        for step in range(n_steps):
            diag = make_diag(n, step + 1)
            expected = b / diag

            print(f"\nstep {step + 1}")
            print_sample("diag", diag)

            x_rebuild, rebuild_elapsed = timed_step(
                "rebuild Mat/PC/KSP + solve",
                rebuild_and_solve,
                diag,
                b,
            )
            x_update, update_elapsed = timed_step(
                "persistent Mat update + solve",
                update_and_solve,
                matrix.handle,
                solver.handle,
                pc.handle,
                diag,
                b,
            )

            rebuild_times.append(rebuild_elapsed)
            update_times.append(update_elapsed)

            print_sample("rebuild x", x_rebuild)
            print_sample("updated persistent x", x_update)
            print_sample("expected", expected)
            print("rebuild/expected norm:", jnp.linalg.norm(x_rebuild - expected))
            print("update/expected norm:", jnp.linalg.norm(x_update - expected))
            print("update/rebuild norm:", jnp.linalg.norm(x_update - x_rebuild))

            np.testing.assert_allclose(np.asarray(x_rebuild), np.asarray(expected), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(np.asarray(x_update), np.asarray(expected), rtol=1e-10, atol=1e-10)

        rebuild_total = sum(rebuild_times)
        update_total = sum(update_times)
        print("\nsummary")
        print(f"rebuild total: {rebuild_total:.6f} s")
        print(f"persistent update total: {update_total:.6f} s")
        print(f"persistent/rebuild ratio: {update_total / rebuild_total if rebuild_total else np.inf}")
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


if __name__ == "__main__":
    main()
