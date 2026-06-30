"""Smoke test for the split Nathan/contact PETSc bridge prototype.

This exercises the intended path:

    Residual/Jacobian callables
        -> evaluated JAX objects at x0
        -> PETSc assembled Mat/PC/KSP solve

Run from the repository root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v7/testNathanContactPETScBridge.py
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from fe_jax.petsc_backend import (
    Residual,
    Jacobian,
    JacobianDiagonal,
    PETScKSPOptions,
    PETScKSPType,
    PETScPCType,
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


def main():
    n = 8
    diag = jnp.array([4.0, 3.0, 9.0, 3.0, 4.0, 8.0, 6.0, 4.0])
    b = jnp.ones((n,), dtype=diag.dtype)
    x_0 = jnp.zeros_like(b)
    constraints = IdentityConstraints()

    residual = Residual(residual_fn, dirichlet_bcs_builtin=True)
    jacobian = Jacobian(jacobian_fn, dirichlet_bcs_builtin=True)
    jacobian_diagonal = JacobianDiagonal(jacobian_diagonal_fn, dirichlet_bcs_builtin=True)
    options = PETScKSPOptions(ksp_type=PETScKSPType.LGMRES, pc_type=PETScPCType.JACOBI)

    print("Testing callable -> JAX objects -> PETSc solve.")
    x = nathan_callable_linear_solve_petsc_branch(
        residual,
        jacobian,
        jacobian_diagonal,
        constraints,
        x_0,
        diag,
        b,
        options=options,
    )

    expected = b / diag
    print("x:", np.asarray(x))
    print("expected:", np.asarray(expected))
    np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=1e-10, atol=1e-10)

    print("Testing explicit callable -> object evaluation entry point.")
    system = build_linear_system_objects(
        residual,
        jacobian,
        jacobian_diagonal,
        constraints,
        x_0,
        diag,
        b,
    )
    x_from_objects = solve_evaluated_petsc_branch(
        system,
        constraints,
        x_0,
        options=options,
    )

    print("x from evaluated objects:", np.asarray(x_from_objects))
    np.testing.assert_allclose(
        np.asarray(x_from_objects),
        np.asarray(expected),
        rtol=1e-10,
        atol=1e-10,
    )


if __name__ == "__main__":
    main()
