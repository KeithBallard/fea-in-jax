import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import buildKSP_Keith
import differentiateKSP_Keith


def jac(_x0):
    n = 8
    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_vals = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)
    return mat_shape, mat_vals, mat_rows, mat_cols


def main():
    shape, vals, rows, cols = jac(None)
    b = jnp.ones((8,), dtype=jnp.float64)

    matrix = buildKSP_Keith.linearMatrixInit(jac=jac, x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)

    try:
        solve = differentiateKSP_Keith.linearSolverSolve

        x = solve(solver, pc, shape, rows, cols, vals, b)
        expected_x = 1.0 / vals

        jac_b = jax.jacfwd(lambda rhs: solve(solver, pc, shape, rows, cols, vals, rhs))(b)
        expected_jac_b = jnp.diag(1.0 / vals)

        grad_vals = jax.grad(lambda v: jnp.sum(solve(solver, pc, shape, rows, cols, v, b)))(vals)
        expected_grad_vals = -1.0 / (vals * vals)

        print("x:", x)
        print("expected x:", expected_x)
        print("jacfwd wrt b:", jac_b)
        print("expected jacfwd wrt b:", expected_jac_b)
        print("grad wrt vals:", grad_vals)
        print("expected grad wrt vals:", expected_grad_vals)

        assert jnp.allclose(x, expected_x)
        assert jnp.allclose(jac_b, expected_jac_b)
        assert jnp.allclose(grad_vals, expected_grad_vals)
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


if __name__ == "__main__":
    main()
