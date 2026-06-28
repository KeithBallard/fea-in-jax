"""Timing comparison for the Nathan/contact PETSc bridge prototype.

This compares:

    1. Nathan-style callable wrappers -> evaluated JAX objects -> PETSc solve
    2. Already-evaluated JAX objects -> PETSc solve
    3. Regular JAX CG using a diagonal matvec
    4. Direct JAX diagonal solve, as a lower-bound sanity baseline

Run from the repository root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v7/testNathanContactPETScBridgeTiming.py 1000 5
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
    JacobianDiagonal,
    PETScKSPOptions,
    PETScKSPType,
    PETScPCType,
    Residual,
    build_linear_system_objects,
    nathan_callable_linear_solve_petsc_branch,
    solve_evaluated_petsc_branch,
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


def make_diag(n):
    indices = jnp.arange(n, dtype=jnp.float64)
    return 3.0 + jnp.mod(17.0 * indices + 2.0, 29.0)


def diagonal_coo(diag):
    indices = jnp.arange(diag.shape[0], dtype=jnp.int32)
    return jsparse.COO(
        (diag, indices, indices),
        shape=(diag.shape[0], diag.shape[0]),
        rows_sorted=True,
        cols_sorted=True,
    )


def residual_fn(x, diag, b):
    return diag * x - b


def jacobian_fn(x, diag, b):
    del x, b
    return diagonal_coo(diag)


def jacobian_diagonal_fn(x, diag, b):
    del x, b
    return diag


@jax.jit
def jax_cg_solve(diag, b):
    matvec = lambda x: diag * x
    x, _ = jspla.cg(
        A=matvec,
        b=b,
        tol=1e-14,
        atol=1e-10,
        maxiter=100000,
    )
    return x


@jax.jit
def jax_direct_diagonal_solve(diag, b):
    return b / diag


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
    return result


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    diag = make_diag(n)
    b = jnp.ones((n,), dtype=diag.dtype)
    x_0 = jnp.zeros_like(b)
    constraints = IdentityConstraints()

    residual = Residual(residual_fn, dirichlet_bcs_builtin=True)
    jacobian = Jacobian(jacobian_fn, dirichlet_bcs_builtin=True)
    jacobian_diagonal = JacobianDiagonal(jacobian_diagonal_fn, dirichlet_bcs_builtin=True)
    options = PETScKSPOptions(ksp_type=PETScKSPType.LGMRES, pc_type=PETScPCType.JACOBI)

    print(f"diagonal size: {n}")
    print(f"timing repeats: {repeats}")
    print("diag first 10:", np.asarray(diag[:10]))

    print("\nWarming up all paths.")
    expected = block(jax_direct_diagonal_solve(diag, b))
    block(jax_cg_solve(diag, b))
    block(
        nathan_callable_linear_solve_petsc_branch(
            residual,
            jacobian,
            jacobian_diagonal,
            constraints,
            x_0,
            diag,
            b,
            options=options,
        )
    )
    system = build_linear_system_objects(
        residual,
        jacobian,
        jacobian_diagonal,
        constraints,
        x_0,
        diag,
        b,
    )
    block(solve_evaluated_petsc_branch(system, constraints, x_0, options=options))

    print("\nTiming solve paths.")
    x_callable = time_call(
        "PETSc bridge callable -> objects -> solve",
        repeats,
        lambda: nathan_callable_linear_solve_petsc_branch(
            residual,
            jacobian,
            jacobian_diagonal,
            constraints,
            x_0,
            diag,
            b,
            options=options,
        ),
    )
    x_evaluated = time_call(
        "PETSc bridge evaluated objects -> solve",
        repeats,
        lambda: solve_evaluated_petsc_branch(system, constraints, x_0, options=options),
    )
    x_cg = time_call("JAX scipy CG", repeats, lambda: jax_cg_solve(diag, b))
    x_direct = time_call("JAX direct diagonal", repeats, lambda: jax_direct_diagonal_solve(diag, b))

    print("\nCorrectness checks.")
    print("expected first 10:", np.asarray(expected[:10]))
    print("PETSc callable first 10:", np.asarray(x_callable[:10]))
    print("PETSc evaluated first 10:", np.asarray(x_evaluated[:10]))
    print("JAX CG first 10:", np.asarray(x_cg[:10]))
    print("JAX direct first 10:", np.asarray(x_direct[:10]))

    np.testing.assert_allclose(np.asarray(x_callable), np.asarray(expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_evaluated), np.asarray(expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_cg), np.asarray(expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(x_direct), np.asarray(expected), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    main()
