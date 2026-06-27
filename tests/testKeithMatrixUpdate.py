import jax

jax.config.update("jax_enable_x64", True)

import cupy as cp
import jax.numpy as jnp

import buildKSP_Keith


def jac(_x0):
    n = 8
    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_vals = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)
    return mat_shape, mat_vals, mat_rows, mat_cols


def main():
    shape, vals_initial, rows, cols = jac(None)
    vals_updated = vals_initial.at[0].set(100.0)

    matrix = buildKSP_Keith.linearMatrixInit(jac=jac, x0=None)
    matrix.handle.block_until_ready()

    try:
        raw_handle = cp.asarray(matrix.handle)
        mat_before = buildKSP_Keith.__retrieve_MAT(raw_handle)

        print("matrix handle before update:", matrix.handle)
        print("PETSc Mat handle before update:", mat_before.handle)
        print("initial vals:", vals_initial)

        matrix_after_update = buildKSP_Keith.linearMatrixUpdate(matrix, vals_updated)
        matrix_after_update.handle.block_until_ready()

        raw_handle_after = cp.asarray(matrix_after_update.handle)
        mat_after = buildKSP_Keith.__retrieve_MAT(raw_handle_after)

        print("matrix handle after update:", matrix_after_update.handle)
        print("PETSc Mat handle after update:", mat_after.handle)
        print("updated vals:", vals_updated)
        print("same store handle:", int(raw_handle) == int(raw_handle_after))
        print("same PETSc Mat object:", mat_before.handle == mat_after.handle)
    finally:
        buildKSP_Keith.linearMatrixCleanup(matrix)


if __name__ == "__main__":
    main()
