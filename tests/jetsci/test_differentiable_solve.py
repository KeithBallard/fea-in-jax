import jax
import jax.numpy as jnp

from options import *
from solve import *

jax.config.update("jax_enable_x64", True)

def matrix_from_phi(phi):
    phi = jnp.asarray(phi, dtype=jnp.float64)
    n = phi.shape[0]
    diag = 4.0 + 0.10 * phi
    A = jnp.diag(diag)
    A = A + 0.20 * jnp.diag(jnp.ones(n - 1, dtype=jnp.float64), k=1)
    A = A + 0.15 * jnp.diag(jnp.ones(n - 1, dtype=jnp.float64), k=-1)
    return A


def residual(phi, x):
    return matrix_from_phi(phi) @ x - phi


def solve_with_petsc(phi, x0, options):
    x, _ = differentiable_solve(
        options,
        residual,
        None,
        x0,
        phi,
    )
    return x

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

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_precond_type=PETScPreconditionerType.JACOBI,
        linear_solve_type=PETScLinearSolverType.LGMRES,
    )

    phi = jnp.array([4.0, 9.0, 16.0],dtype=jnp.float64)
    phi_2 = jnp.array([-3.0, 5.0, 100.0],dtype=jnp.float64)

    sol, _ = differentiable_solve(
        options,
        residual,
        None,
        jnp.array([1.5, 2.5, 3.5],dtype=jnp.float64),
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
    print(f"Residual (x): {residual(phi, sol)}")

    @jax.custom_jvp
    def solve_only(active_phi):
        return differentiable_solve(
            options,
            residual,
            None,
            jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64),
            active_phi,
        )[0]

    @solve_only.defjvp
    def _solve_only_jvp(primals, tangents):
        (active_phi,), (phi_dot,) = primals, tangents
        x_star = solve_only(active_phi)
        x_dot = jnp.linalg.solve(J_x_expected, phi_dot)
        return x_star, x_dot

    jac_fwd = jax.jacfwd(solve_only)(phi_2)
    jac_rev = jax.jacrev(solve_only)(phi_2)
    expected_jac = jnp.linalg.inv(J_x_expected)

    print("Jacobian of solve wrt phi (jacfwd):")
    print(jac_fwd)
    print("Jacobian of solve wrt phi (jacrev):")
    print(jac_rev)
    print("Expected Jacobian of solve wrt phi:")
    print(expected_jac)

if __name__ == "__main__":
    main()
