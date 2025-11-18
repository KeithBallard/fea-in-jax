import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from jax.experimental.buffer_callback import buffer_callback

import scipy.sparse

import pypardiso
import cupy as cp

import numpy as np

def residual(x: jnp.ndarray):
    return jnp.array(
        [
            2.0 * x[0] + x[1] + x[2] - 1.0,
            x[0] - x[1] + x[2] - 2.0,
            x[2] - 1.0,
        ],
    )

def __pypardiso_solve_impl(
    ctx,
    out,
    A_data: jnp.ndarray,
    A_row: jnp.ndarray,
    A_col: jnp.ndarray,
    b: jnp.ndarray,
):
    print("A.row", cp.asnumpy(cp.asarray(A_row)))
    # print("A.col", cp.asnumpy(cp.asarray(A.col)))
    # print("A.data", cp.asnumpy(cp.asarray(A.data)))
    A_scipy = scipy.sparse.csr_matrix(
        (
            cp.asarray(A_data).get().astype(np.float64),
            (cp.asarray(A_row).get().astype(np.int32), cp.asarray(A_col).get().astype(np.int32)),
        ),
        shape=(b.shape[0], b.shape[0]),
    )
    print(A_scipy.indices.dtype)
    b_np = cp.asarray(b).get().astype(np.float64)
    result = pypardiso.spsolve(A_scipy, b_np)
    cp.asarray(out)[...] = cp.asarray(result)

@jax.jit
def __pypardiso_solve(A: jsparse.COO, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    #jax.debug.print("A.row - jax {}", A.row)
    #jax.debug.print("A.data - jax {}", A.data)
    return buffer_callback(
            __pypardiso_solve_impl, result_info, command_buffer_compatible=False
        )(A.data, A.row, A.col, b)

x0 = jnp.array([0.0, 0.0, 0.0])
J_jax = jax.jacfwd(residual)(x0)
J_sparse = jsparse.COO.fromdense(J_jax, index_dtype=jnp.int32)

@jax.jit
def solve():
    x = __pypardiso_solve(J_sparse, -residual(x0))
    return x

print(solve())