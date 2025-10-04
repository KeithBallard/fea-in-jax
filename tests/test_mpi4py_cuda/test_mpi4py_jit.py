from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD

@jax.jit
def get_rank() -> jnp.ndarray:
    # NOTE pure_callback is NOT needed since the MPI call simply returns a value and does not take
    # a traced array (which are the troublemakers).
    rank = comm.Get_rank()
    return jnp.array([rank])

@jax.jit
def get_comm_size_rand_array() -> jnp.ndarray:
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        array = jnp.array(np.random.rand(size), dtype=jnp.float32)
    else:
        array = jnp.zeros(shape=(size,), dtype=jnp.float32)    

    # NOTE pure_callback is needed since the MPI bcast call is not understood by the JIT compiler
    result_shape = jax.ShapeDtypeStruct(shape=array.shape, dtype=array.dtype)
    array = jax.pure_callback(comm.bcast, result_shape, array, 0, vmap_method='sequential')
    return array
    
# NOTE the function is JIT'ed with an MPI call, but there are a few caveats:
# 1. Per the docs, data is moved to host memory before being passed to function given to
#    pure_callback. So, if the array was on device memory, it would be moved to host,
#    MPI bcast called on host array, and then moved back to device memory. However,
#    I have not tested it to verify the data is being moved when the function is MPI
#    aware...
# 2. While compatible with JIT, the function is not compatible with jvp unless the
#    derivative information is provided. See:
#    https://docs.jax.dev/en/latest/external-callbacks.html#example-pure-callback-with-custom-jvp
# Refer to the following link for more information:
# https://docs.jax.dev/en/latest/external-callbacks.html
array = get_comm_size_rand_array()
print(get_rank(), array)