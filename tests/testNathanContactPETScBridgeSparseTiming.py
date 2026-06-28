"""Timing comparison on a poorly conditioned sparse matrix.

This uses a 2D variable-coefficient diffusion matrix with a configurable
coefficient contrast. The matrix is SPD, sparse, coupled, and intentionally
less friendly than the uniform Laplacian or diagonal tests.

Run from the repository root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v7/testNathanContactPETScBridgeSparseTiming.py 256 3 1e6
"""

from __future__ import annotations

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.scipy.sparse.linalg as jspla

from fe_jax.petsc_backend import (
    Jacobian,
    PETScKSPOptions,
    PETScKSPType,
    PETScPCType,
    Residual,
    build_linear_system_objects,
    cleanup_persistent_state,
    init_persistent_state,
    nathan_callable_linear_solve_petsc_branch,
    solve_evaluated_petsc_branch,
    solve_with_persistent_state,
)

class IdentityConstraints:
    """Minimal stand-in for fea-in-jax ConstraintSystem with no constrained DOFs."""

    dep_dofs = jnp.array([], dtype=jnp.int32)

    def apply_to_residual(self, residual, x):
        del x
        return residual

    def apply_to_delta_solution(self, delta_x, x_0):
        del x_0
        return delta_x


def _node_coefficient(ix, iy, grid_n, contrast):
    if grid_n <= 1:
        return 1.0
    phase = (ix + 3 * iy) / (4 * (grid_n - 1))
    oscillation = 0.25 * (1.0 + np.sin(0.37 * ix + 0.19 * iy))
    exponent = np.clip(phase + oscillation, 0.0, 1.0)
    return contrast**exponent


def _edge_coefficient(ix0, iy0, ix1, iy1, grid_n, contrast):
    k0 = _node_coefficient(ix0, iy0, grid_n, contrast)
    k1 = _node_coefficient(ix1, iy1, grid_n, contrast)
    return 2.0 * k0 * k1 / (k0 + k1)


def variable_diffusion_2d_coo(grid_n, contrast=1.0e6, mass=1.0e-6):
    row_chunks = []
    col_chunks = []
    val_chunks = []

    for iy in range(grid_n):
        for ix in range(grid_n):
            center = iy * grid_n + ix
            diagonal = mass
            neighbors = []

            if ix > 0:
                weight = _edge_coefficient(ix, iy, ix - 1, iy, grid_n, contrast)
                diagonal += weight
                neighbors.append((center - 1, -weight))
            if ix + 1 < grid_n:
                weight = _edge_coefficient(ix, iy, ix + 1, iy, grid_n, contrast)
                diagonal += weight
                neighbors.append((center + 1, -weight))
            if iy > 0:
                weight = _edge_coefficient(ix, iy, ix, iy - 1, grid_n, contrast)
                diagonal += weight
                neighbors.append((center - grid_n, -weight))
            if iy + 1 < grid_n:
                weight = _edge_coefficient(ix, iy, ix, iy + 1, grid_n, contrast)
                diagonal += weight
                neighbors.append((center + grid_n, -weight))

            row_chunks.append(center)
            col_chunks.append(center)
            val_chunks.append(diagonal)
            for col, value in neighbors:
                row_chunks.append(center)
                col_chunks.append(col)
                val_chunks.append(value)

    rows = jnp.array(row_chunks, dtype=jnp.int32)
    cols = jnp.array(col_chunks, dtype=jnp.int32)
    vals = jnp.array(val_chunks, dtype=jnp.float64)
    size = grid_n * grid_n
    return jsparse.COO(
        (vals, rows, cols),
        shape=(size, size),
        rows_sorted=True,
        cols_sorted=False,
    )


def residual_fn(x, matrix, b):
    return matrix @ x - b


def jacobian_fn(x, matrix, b):
    del x, b
    return matrix


@jax.jit
def jax_cg_solve(matrix, b):
    x, _ = jspla.cg(
        A=lambda v: matrix @ v,
        b=b,
        tol=1e-12,
        atol=1e-10,
        maxiter=100000,
    )
    return x


def block(value):
    return jax.block_until_ready(value)


def time_call(label, repeats, fn):
    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = block(fn())
        elapsed.append(time.perf_counter() - start)

    arr = np.asarray(elapsed)
    print(
        f"{label}: min={arr.min():.6f} s, "
        f"mean={arr.mean():.6f} s, "
        f"last={arr[-1]:.6f} s"
    )
    return result, {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "last": float(arr[-1]),
    }


def print_timing_summary(timings):
    print("\nTiming summary")
    print("--------------")
    print(f"{'path':45s} {'min (s)':>10s} {'mean (s)':>10s} {'last (s)':>10s}")
    for label, values in timings.items():
        print(
            f"{label:45s} "
            f"{values['min']:10.6f} "
            f"{values['mean']:10.6f} "
            f"{values['last']:10.6f}"
        )

    if "PETSc persistent solve only" in timings and "JAX scipy CG" in timings:
        ratio = timings["PETSc persistent solve only"]["mean"] / timings["JAX scipy CG"]["mean"]
        print(f"\nPETSc persistent / JAX CG mean ratio: {ratio:.3f}x")

    if "PETSc evaluated rebuild" in timings and "PETSc persistent solve only" in timings:
        overhead = timings["PETSc evaluated rebuild"]["mean"] - timings["PETSc persistent solve only"]["mean"]
        print(f"PETSc rebuild overhead over persistent mean: {overhead:.6f} s")


def main():
    grid_n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    contrast = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e6
    size = grid_n * grid_n

    matrix = variable_diffusion_2d_coo(grid_n, contrast=contrast)
    b = jnp.ones((size,), dtype=matrix.data.dtype)
    x_0 = jnp.zeros_like(b)
    constraints = IdentityConstraints()

    residual = Residual(residual_fn, dirichlet_bcs_builtin=True)
    jacobian = Jacobian(jacobian_fn, dirichlet_bcs_builtin=True)
    options = PETScKSPOptions(ksp_type=PETScKSPType.CG, pc_type=PETScPCType.JACOBI)

    print(f"grid size: {grid_n} x {grid_n}")
    print(f"matrix size: {size}")
    print(f"nnz: {matrix.nse}")
    print(f"coefficient contrast: {contrast:g}")
    print(f"timing repeats: {repeats}")

    print("\nWarming up all paths.")
    x_cg_expected = block(jax_cg_solve(matrix, b))
    block(
        nathan_callable_linear_solve_petsc_branch(
            residual,
            jacobian,
            None,
            constraints,
            x_0,
            matrix,
            b,
            options=options,
        )
    )
    system = build_linear_system_objects(
        residual,
        jacobian,
        None,
        constraints,
        x_0,
        matrix,
        b,
    )
    block(solve_evaluated_petsc_branch(system, constraints, x_0, options=options))

    persistent_state = init_persistent_state(system.J_sparse, options=options)
    try:
        block(solve_with_persistent_state(persistent_state, system.J_sparse, system.rhs, update_matrix=False))
        block(solve_with_persistent_state(persistent_state, system.J_sparse, system.rhs, update_matrix=True))

        print("\nTiming solve paths.")
        timings = {}
        x_callable, timings["PETSc callable rebuild"] = time_call(
            "PETSc bridge callable -> objects -> rebuild solve",
            repeats,
            lambda: nathan_callable_linear_solve_petsc_branch(
                residual,
                jacobian,
                None,
                constraints,
                x_0,
                matrix,
                b,
                options=options,
            ),
        )
        x_evaluated, timings["PETSc evaluated rebuild"] = time_call(
            "PETSc bridge evaluated objects -> rebuild solve",
            repeats,
            lambda: solve_evaluated_petsc_branch(system, constraints, x_0, options=options),
        )
        x_persistent_no_update, timings["PETSc persistent solve only"] = time_call(
            "PETSc persistent Mat/PC/KSP -> solve only",
            repeats,
            lambda: solve_with_persistent_state(
                persistent_state,
                system.J_sparse,
                system.rhs,
                update_matrix=False,
            ),
        )
        x_persistent_update, timings["PETSc persistent update"] = time_call(
            "PETSc persistent Mat update -> solve",
            repeats,
            lambda: solve_with_persistent_state(
                persistent_state,
                system.J_sparse,
                system.rhs,
                update_matrix=True,
            ),
        )
        x_cg, timings["JAX scipy CG"] = time_call("JAX scipy CG", repeats, lambda: jax_cg_solve(matrix, b))

        print("\nCorrectness checks against JAX CG.")
        print("JAX CG first 10:", np.asarray(x_cg[:10]))
        print("PETSc callable first 10:", np.asarray(x_callable[:10]))
        print("PETSc evaluated first 10:", np.asarray(x_evaluated[:10]))
        print("PETSc persistent solve first 10:", np.asarray(x_persistent_no_update[:10]))
        print("PETSc persistent update first 10:", np.asarray(x_persistent_update[:10]))

        np.testing.assert_allclose(np.asarray(x_callable), np.asarray(x_cg_expected), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(np.asarray(x_evaluated), np.asarray(x_cg_expected), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(np.asarray(x_persistent_no_update), np.asarray(x_cg_expected), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(np.asarray(x_persistent_update), np.asarray(x_cg_expected), rtol=1e-7, atol=1e-7)

        print_timing_summary(timings)
    finally:
        cleanup_persistent_state(persistent_state)


if __name__ == "__main__":
    main()
