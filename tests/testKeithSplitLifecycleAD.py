import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import buildKSP_Keith


def jac(x0):
    n = x0.shape[0]
    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_vals = x0
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)
    return mat_shape, mat_vals, mat_rows, mat_cols


def split_lifecycle_only_loss(x0):
    matrix = buildKSP_Keith.linearMatrixInit(jac=jac, x0=x0)
    matrix = buildKSP_Keith.linearMatrixUpdate(matrix, x0 + 1.0)
    pc = buildKSP_Keith.linearPCInit(matrix)
    ksp = buildKSP_Keith.linearKSPInit(matrix)

    buildKSP_Keith.linearSolverCleanup(ksp)
    buildKSP_Keith.linearPCCleanup(pc)
    buildKSP_Keith.linearMatrixCleanup(matrix)
    return jnp.sum(x0 * x0)


def main():
    x0 = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    grad_x0 = jax.grad(split_lifecycle_only_loss)(x0)
    expected = 2.0 * x0

    print("grad through split lifecycle-only function:", grad_x0)
    print("expected:", expected)
    assert jnp.allclose(grad_x0, expected)


if __name__ == "__main__":
    main()
