from src.fe_jax import contact
from jax import numpy as jnp
import numpy as np

eps = 0.1

fiber0 = jnp.array([[-1,0,0],[0,0,0],[1,0,0]],dtype = np.float64)
fiber1 = jnp.array([[0,eps,-1],[0,eps,0],[0,eps,1]], dtype = np.float64)
fiber2 = jnp.array([[0,-eps,-1],[0,-eps,0],[0,-eps,1]], dtype = np.float64)

fibers = [fiber0,fiber1,fiber2]
ct = contact.contact_batch(
    fibers = fibers,
    radius = 0.12,
    distinct_fiber_fn = contact.distinct_fiber_node2node,
    self_fiber_fn = contact.self_fiber_node2node,
)
