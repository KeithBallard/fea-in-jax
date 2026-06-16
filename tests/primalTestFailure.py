import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from primitiveKSP import solve_from_coo


def main():
    n = 8
    diag = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    b = jnp.ones((n,), dtype=jnp.float64)

    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)

    def solve_with_vals(vals):
        return solve_from_coo(mat_shape, mat_rows, mat_cols, vals, b)

    print("Testing direct VJP with respect to matrix values.")
    try:
        x, pullback = jax.vjp(solve_with_vals, diag)
        vals_bar, = pullback(jnp.ones_like(x))

        expected_x = 1.0 / diag
        expected_vals_bar = -1.0 / (diag * diag)

        print("direct vjp succeeded")
        print("x:", x)
        print("expected x:", expected_x)
        print("vals_bar:", vals_bar)
        print("expected vals_bar:", expected_vals_bar)

        assert jnp.allclose(x, expected_x)
        assert jnp.allclose(vals_bar, expected_vals_bar)
    except NotImplementedError as err:
        print("direct vjp hit expected current limitation:")
        print(err)

    print("Testing jacrev with respect to RHS.")
    try:
        jacrev_b = jax.jacrev(
            lambda rhs: solve_from_coo(mat_shape, mat_rows, mat_cols, diag, rhs)
        )(b)
        expected_jacrev_b = jnp.diag(1.0 / diag)

        print("jacrev wrt b succeeded")
        print("jacrev wrt b:", jacrev_b)
        print("expected jacrev wrt b:", expected_jacrev_b)

        assert jnp.allclose(jacrev_b, expected_jacrev_b)
    except NotImplementedError as err:
        print("jacrev wrt b hit expected current limitation:")
        print(err)

    print("Testing jacrev with respect to matrix values.")
    try:
        jacrev_vals = jax.jacrev(solve_with_vals)(diag)
        expected_jacrev_vals = -jnp.diag(1.0 / (diag * diag))

        print("jacrev wrt vals succeeded")
        print("jacrev wrt vals:", jacrev_vals)
        print("expected jacrev wrt vals:", expected_jacrev_vals)

        assert jnp.allclose(jacrev_vals, expected_jacrev_vals)
    except NotImplementedError as err:
        print("jacrev wrt vals hit expected current limitation:")
        print(err)


if __name__ == "__main__":
    main()
