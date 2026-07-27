import jax
import jax.numpy as jnp
import warp as wp
import numpy as np

# Initialize warp
wp.init()

# Create a JAX array on GPU (if available) or CPU
x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

# Convert to Warp array
try:
    # Warp 1.0+ supports converting JAX arrays using DLPack or standard array protocol
    wp_x = wp.array(x, copy=False)
    print("Direct wp.array conversion successful!")
    print("Warp array content:", wp_x)
except Exception as e:
    print("Direct conversion failed, trying cupy/numpy bridge:", e)
    # Fallback via cupy/numpy depending on backend
    import cupy as cp
    cp_x = cp.asarray(x)
    wp_x = wp.array(cp_x, copy=False)
    print("CuPy bridge conversion successful!")
    print("Warp array content:", wp_x)
