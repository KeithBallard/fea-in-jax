# TODO write a test case that tests using JAX jvp for a simple polynomial for which the exact answer is known

# System of equations:
# 2x -  y -  z =  1
# -x + 2y -  z =  4
# -x -  y + 2z = -1

# With the additional constraint that z = 1, the system is:
# 2x -  y = 2
# -x + 2y = 5
# Solution: x = 3, y = 4

# Without constraint:
# Residual = [ 2x +  y + z - 1 ]
#            [  x -  y + z - 2 ]
#            [ 3x - 2y - z + 1 ]

# With constraint:
# Residual = [ 2x -  y - 2 ]
#            [ -x + 2y - 5 ]
#            [ z - 1 ]
# Jacobian = [ 2 -1  0 ]
#            [-1  2  0 ]
#            [ 0  0  1 ]

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from helper import *
from fe_jax.solve_cg import *

from enum import Enum


class SolverType(Enum):
    CG = 0
    CG_JACOBI = 1
    CG_INV = 2


dtype = jnp.float32


# Note: z = 1.0
def residual(x: jnp.ndarray):
    return jnp.array(
        [
            2.0 * x[0] - x[1] - 2.0,
            -x[0] + 2.0 * x[1] - 5.0,
            x[2] - 1.0,
        ],
        dtype=dtype,
    )


x0 = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
b = jnp.array([2.0, 5.0, 1.0], dtype=dtype)
tol = 1e-12
atol = 0.0

J_jax = jax.jacfwd(residual)(x0)
print("J(x) = \n", J_jax)

print(jnp.dot(J_jax, x0))

for solver_type in SolverType:

    match solver_type:
        case SolverType.CG:
            M = None
        case SolverType.CG_JACOBI:
            M = jnp.diag(1.0 / jnp.diag(J_jax))
        case SolverType.CG_INV:
            M = jnp.linalg.inv(J_jax)
    print("M(x) = \n", M)

    solver = cg
    # solver = jax.scipy.sparse.linalg.cg
    x_solution, info = solver(
        A=J_jax, b=b, M=M, tol=jnp.astype(tol, dtype), atol=jnp.astype(atol, dtype)
    )

    print("x = ", x_solution)
    r_solution = residual(x_solution)
    print("R = ", r_solution)

    print(info["iterations"])
    x_iter = jnp.linspace(
        0, info["iterations"], info["iterations"] + 1, dtype=jnp.int32
    )
    y_r_norm = info["residual_norm_history"][0 : info["iterations"] + 1]

    print(x_iter, y_r_norm)
    plt.plot(x_iter, y_r_norm)
    plt.title(f"Residual History During Iteration Using {solver_type}")
    plt.xlabel("iteration")
    plt.ylabel("|R|")
    plt.yscale("log")
    plt.show()
