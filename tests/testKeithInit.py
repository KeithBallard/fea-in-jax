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
    shape, vals, rows, cols = jac(None)
    matrix = buildKSP_Keith.linearMatrixInit(jac=jac, x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    ksp = buildKSP_Keith.linearKSPInit(matrix)

    try:
        print("matrix handle:", matrix.handle)
        print("pc handle:", pc.handle)
        print("ksp handle:", ksp.handle)

        b = jnp.ones((8,), dtype=jnp.float64)
        x = runKSP_Keith.__petsc_solve(ksp, pc, b)
        print("solve rhs:", b)
        print("solve result:", x)
    finally:
        buildKSP_Keith.linearSolverCleanup(ksp)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)
        print("linearSolverCleanup/linearPCCleanup/linearMatrixCleanup completed")


if __name__ == "__main__":
    main()
