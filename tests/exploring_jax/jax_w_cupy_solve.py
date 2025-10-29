# Residual = [ 2x +  y + z - 1 ]
#            [  x -  y + z - 2 ]
#            [ z - z0 ]
# Jacobian = [ 2  1  1 ]
#            [ 1 -1  1 ]
#            [ 0  0  1 ]

import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import buffer_callback
from jax.dlpack import from_dlpack

jax.config.update("jax_enable_x64", True)

import cupy as cp
import jaxbind

dtype = jnp.float64

# Note: z0 = 1.0
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

b = jnp.array([1.0, 2.0, 1.0], dtype=dtype)

def cupy_sin(ctx, out, x):
    cp.asarray(out)[...] = cp.sin(cp.asarray(x))

def build_sin(x):
    out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return buffer_callback(cupy_sin, out_type)

@jax.jit
def sin_jit(x: jnp.ndarray):
    sin_x = build_sin(x)
    return sin_x(x)

x_solution = sin_jit(b)

print("x = ", x_solution)
r_solution = residual(x_solution)
print('R = ', r_solution)
#assert jnp.linalg.norm(r_solution) < 1e-12
