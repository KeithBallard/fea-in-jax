# TODO write a test case that tests using JAX jvp for a simple polynomial for which the exact answer is known

# Residual = [ 2x +  y + z - 1 ]
#            [  x -  y + z - 2 ]
#            [ 3x - 2y - z + 1 ]
# Jacobian = [ 2  1  1 ]
#            [ 1 -1  1 ]
#            [ 3 -2 -1 ]

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def sweep_solver_tols(dtype, r_ax):
    def residual(x: jnp.ndarray):
        return jnp.array(
            [
                2.0 * x[0] + x[1] + x[2] - 1.0,
                x[0] - x[1] + x[2] - 2.0,
                3.0 * x[0] - 2.0 * x[1] - x[2] + 1.0,
            ],
            dtype=dtype,
        )

    x = jnp.array([1.0, 1.0, 1.0], dtype=dtype)
    z = jnp.array([1.0, 1.0, 1.0], dtype=dtype)
    result = jax.jvp(residual, (x,), (z,))
    print("R(x) = ", result[0])
    print("J(x) * z = ", result[1])

    J = jnp.array([[2.0, 1.0, 1.0], [1.0, -1.0, 1.0], [3.0, -2.0, -1.0]], dtype=dtype)
    assert jnp.isclose(jnp.dot(J, z), result[1]).all()

    J_jax = jax.jacfwd(residual)(x)
    print("J(x) = \n", J_jax)

    tols = np.logspace(-10, -1)
    r_norms = np.zeros_like(tols)
    print(tols)
    for i, tol in enumerate(tols):
        print("Running tol = ", tol)

        b = jnp.array([1.0, 2.0, -1.0], dtype=dtype)
        atol = 0.

        x_solution, info = jax.scipy.sparse.linalg.bicgstab(
            A=J_jax, b=b, tol=jnp.astype(tol, dtype), atol=jnp.astype(atol, dtype)
        )

        print("  x = ", x_solution)
        r_solution = residual(x_solution)
        r_norms[i] = jnp.linalg.norm(r_solution)
        print("  |R| = ", r_norms[i])

    r_ax.plot(tols, r_norms)
    r_ax.set_xscale("log")
    r_ax.set_yscale("log")
    r_ax.set_xlabel("Relative Tolerance")
    r_ax.set_ylabel("Resulting |R|")
    r_ax.set_title(f"Solution residual vs tolerance for {dtype}")

fig, (r_ax1, r_ax2, r_ax3) = plt.subplots(1,3)
sweep_solver_tols(jnp.float16, r_ax1)
sweep_solver_tols(jnp.float32, r_ax2)
jax.config.update("jax_enable_x64", True)
sweep_solver_tols(jnp.float64, r_ax3)
plt.show()