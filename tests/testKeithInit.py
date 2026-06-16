import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import buildKSP_Keith
import runKSP_Keith

from petsc4py import PETSc


def jac(_x0):
    n = 8
    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_vals = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)
    return mat_shape, mat_vals, mat_rows, mat_cols


def main():

    solver = buildKSP_Keith.linearSolverInit(
        jac=jac,
        res=None,
        diag=None,
        x0=None,
        constructionOptions=None,
    )
    print("linearSolverInit returned:", solver)
    print("solver handle:", solver.handle)

    b = jnp.ones((8,), dtype=jnp.float64)
    x = runKSP_Keith.__petsc_solve(solver, b)
    print("solve rhs:", b)
    print("solve result:", x)

    buildKSP_Keith.linearSolverCleanup(solver)
    print("linearSolverCleanup completed")


if __name__ == "__main__":
    main()
