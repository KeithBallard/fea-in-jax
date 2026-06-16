import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from functools import partial

from primitiveKSP import linear_solve, petsc_cleanup, petsc_init


@jax.jit
def jitted_change(vals):
    return vals.at[0].set(100.0)


@partial(jax.jit, donate_argnums=(0,))
def donated_jitted_change(vals):
    return vals.at[0].set(100.0)


def run_case(name, change_fn):
    n = 8
    diag_initial = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    b = jnp.ones((n,), dtype=jnp.float64)

    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)

    solver = petsc_init(mat_shape, diag_initial, mat_rows, mat_cols)

    try:
        x_before = linear_solve(solver, b)
        diag_changed = change_fn(diag_initial)
        diag_changed.block_until_ready()
        x_after = linear_solve(solver, b)

        expected_initial = 1.0 / jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
        expected_changed = 1.0 / diag_changed

        print()
        print("===", name, "===")
        print("diag_changed:", diag_changed)
        print("x_before:", x_before)
        print("x_after same KSP:", x_after)
        print("expected if KSP copied initial values:", expected_initial)
        print("expected if KSP aliases changed values:", expected_changed)
    finally:
        petsc_cleanup(solver)


def main():
    run_case("ordinary JAX update", lambda vals: vals.at[0].set(100.0))
    run_case("jitted JAX update", jitted_change)
    run_case("donated jitted JAX update", donated_jitted_change)


if __name__ == "__main__":
    main()
