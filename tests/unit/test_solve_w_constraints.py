# TODO write a test case that tests using JAX jvp for a simple polynomial for which the exact answer is known

# System of equations:
# 2x + y + z = 1
# x - y + z = 2
# 3x - 2y - z = -1

# With the additional constraint that z = 1

# Without constraint:
# Residual = [ 2x +  y + z - 1 ]
#            [  x -  y + z - 2 ]
#            [ 3x - 2y - z + 1 ]
# Jacobian = [ 2  1  1 ]
#            [ 1 -1  1 ]
#            [ 3 -2 -1 ]

# With constraint:
# Residual = [ 2x +  y + z - 1 ]
#            [  x -  y + z - 2 ]
#            [ z - z0 ]
# Jacobian = [ 2  1  1 ]
#            [ 1 -1  1 ]
#            [ 0  0  1 ]

from helper import *

import matplotlib.pyplot as plt
from jaxopt import linear_solve

dtype = jnp.float64

# Note: z0 = 1.0
def residual(x: jnp.ndarray):
    return jnp.array(
        [
            2.0 * x[0] + x[1] + x[2] - 1.0,
            x[0] - x[1] + x[2] - 2.0,
            x[2] - 1.0,
        ],
        dtype=dtype,
    )


def test_gmres_and_spsolve_solve_constrained_system():
    x0 = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
    b = jnp.array([1.0, 2.0, 1.0], dtype=dtype)
    tol = 1e-12
    atol = 0.0

    J_jax = jax.jacfwd(residual)(x0)
    print("J(x) = \n", J_jax)

    print(jnp.dot(J_jax, x0))

    M = jnp.diag(1.0 / jnp.diag(J_jax))
    print("M(x) = \n", M)

    x_solution, info = jax.scipy.sparse.linalg.gmres(
        A=J_jax, b=b, M=M, tol=jnp.astype(tol, dtype), atol=jnp.astype(atol, dtype)
    )
    # x_solution = linear_solve.solve_gmres(J_jax, b)

    print("x = ", x_solution)
    r_solution = residual(x_solution)
    print("R = ", r_solution)
    assert jnp.linalg.norm(r_solution) < 1e-12

    import jax.experimental.sparse as jsparse
    import scipy.sparse
    import scipy.sparse.linalg

    J_sparse = jsparse.COO.fromdense(J_jax, index_dtype=jnp.int64)
    J_scipy = scipy.sparse.coo_matrix(
        (
            np.asarray(J_sparse.data),
            (np.asarray(J_sparse.row), np.asarray(J_sparse.col)),
        ),
        shape=J_sparse.shape,
    ).tocsr()
    x_solution_2 = jnp.asarray(scipy.sparse.linalg.spsolve(J_scipy, np.asarray(b)))
    print("x = ", x_solution_2)
    assert jnp.isclose(x_solution, x_solution_2, rtol=1e-12, atol=1e-4).all()

    r_solution = residual(x_solution_2)
    print("R = ", r_solution)
    assert jnp.linalg.norm(r_solution) < 1e-12
