"""
Exploring how to bridge JAX JIT'ed functions and CUPY calls.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from jax.experimental.buffer_callback import buffer_callback
from jax.dlpack import from_dlpack

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cplinalg

##################################################################################################
# Calling CUPY sin (one arg, one output) within JAX JIT function


@jax.jit
def cupy_sin(x):

    def cupy_sin_kernel(ctx, out, x):
        cp.asarray(out)[...] = cp.sin(cp.asarray(x))

    out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    cupy_sin_callback = buffer_callback(cupy_sin_kernel, out_type)

    return cupy_sin_callback(x)


@jax.jit
def sin_jit(x: jnp.ndarray):
    return cupy_sin(jnp.cos(x))


b = jnp.array([1.0, 2.0, 1.0], dtype=dtype)
result = sin_jit(b)
assert jnp.isclose(result, jnp.sin(jnp.cos(b))).all()


##################################################################################################
# Calling CUPY solve (multiple args, one output) within JAX JIT function


# Residual = [ 2x +  y + z - 1 ]
#            [  x -  y + z - 2 ]
#            [ z - z0 ]
# Jacobian = [ 2  1  1 ]
#            [ 1 -1  1 ]
#            [ 0  0  1 ]
@jax.jit
def residual(x: jnp.ndarray):
    return jnp.array(
        [
            2.0 * x[0] + x[1] + x[2] - 1.0,
            x[0] - x[1] + x[2] - 2.0,
            x[2] - 1.0,
        ],
        dtype=dtype,
    )


@jax.jit
def cupy_solve(A, b):

    def kernel(ctx, out, A, b):
        cp.asarray(out)[...] = cp.linalg.solve(cp.asarray(A), cp.asarray(b))

    out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
    cupy_callback = buffer_callback(kernel, out_type)
    return cupy_callback(A, b)


x0 = jnp.zeros_like(b)
J = jax.jacfwd(residual)(x0)
x = cupy_solve(J, b)
assert jnp.isclose(residual(x), jnp.zeros_like(x)).all()


##################################################################################################
# Calling CUPY solve given JAX callable residual function


@jax.jit
def cupy_solve_from_R(residual: jax.tree_util.Partial, b: jnp.ndarray, x0: jnp.ndarray):

    def kernel(ctx, out, A, b):
        cp.asarray(out)[...] = cp.linalg.solve(cp.asarray(A), cp.asarray(b))

    out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
    cupy_callback = buffer_callback(kernel, out_type)
    return cupy_callback(jax.jacfwd(residual)(x0), b)


x = cupy_solve_from_R(jax.tree_util.Partial(residual), b, x0)
assert jnp.isclose(residual(x), jnp.zeros_like(x)).all()


##################################################################################################
# Calling CUPY sparse solver


@jax.jit
def cupy_spsolve(A, b):

    def kernel(ctx, out, A: jsparse.CSR, b):
        A_cp = cpsparse.csr_matrix(
            (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)), shape=A.shape
        )
        cp.asarray(out)[...] = cplinalg.spsolve(A_cp, cp.asarray(b))

    out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
    cupy_callback = buffer_callback(kernel, out_type)
    return cupy_callback(A, b)

J_sp = jsparse.csr_fromdense(J)
x = cupy_spsolve(J_sp, b)
assert jnp.isclose(residual(x), jnp.zeros_like(x)).all()
