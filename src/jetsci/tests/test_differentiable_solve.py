import jetsci
import jax.numpy as jnp

def residual(phi, x):
    a = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    return a @ x - phi


def pure_jax_newton_solve(phi, x0):
    del x0
    a = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    return jnp.linalg.solve(a, phi)


def main():

    options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
        linear_solve_type=jetsci.PETScLinearSolverType.LGMRES,
    )

    phi = jnp.array([4.0, 9.0, 16.0])

    sol, _ = jetsci.differentiable_solve(
        options,
        residual,
        None,
        jnp.array([1.5, 2.5, 3.5]),
        phi,
    )
    print("Solved x:", sol)
    print(f"Residual (x): {residual(phi, sol)}")

    J_x_expected = jnp.asarray(
        [
            [4.0, 1.0, 0.5],
            [0.25, 3.0, 1.5],
            [0.75, 0.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    x_expected = jnp.linalg.solve(J_x_expected, phi)
    print("Expected x:", x_expected)
    print(f"Residual (x): {residual(phi, x_expected)}")


if __name__ == "__main__":
    main()
