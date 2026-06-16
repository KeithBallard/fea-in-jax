import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from primitiveKSP import solve_from_coo


def main():
    n = 8
    key = jax.random.PRNGKey(0)

    diag = jax.random.randint(key, shape=(n,), minval=1, maxval=10).astype(jnp.float64)
    b = jnp.ones((n,), dtype=jnp.float64)

    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)
    mat_vals = diag

    x = solve_from_coo(mat_shape, mat_rows, mat_cols, mat_vals, b)
    expected_x = 1.0 / diag

    jac_b = jax.jacfwd(lambda rhs: solve_from_coo(mat_shape, mat_rows, mat_cols, mat_vals, rhs))(b)
    expected_jac_b = jnp.diag(1.0 / diag)

    grad_vals = jax.grad(
        lambda vals: jnp.sum(solve_from_coo(mat_shape, mat_rows, mat_cols, vals, b))
    )(mat_vals)
    expected_grad_vals = -1.0 / (diag * diag)

    print("diag(A):", diag)
    print("b:", b)
    print("x:", x)
    print("expected x:", expected_x)
    print("jacfwd wrt b:", jac_b)
    print("expected jacobian wrt b:", expected_jac_b)
    print("grad wrt vals:", grad_vals)
    print("expected grad wrt vals:", expected_grad_vals)

    assert jnp.allclose(x, expected_x)
    assert jnp.allclose(jac_b, expected_jac_b)
    assert jnp.allclose(grad_vals, expected_grad_vals)


if __name__ == "__main__":
    main()
