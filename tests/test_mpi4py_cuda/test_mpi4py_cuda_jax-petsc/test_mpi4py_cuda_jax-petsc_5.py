from mpi4py import MPI
import mpi4jax
import jax
import jax.numpy as jnp

# 1. Initialize MPI communicator via mpi4py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 2. Define a JAX function using mpi4jax
@jax.jit
def parallel_sum(x):
    # Pass the 'comm' object to mpi4jax
    packedObject = mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)
    return packedObject

# 3. Execute
data = jnp.ones((4, 4)) * rank
result = parallel_sum(data)

print(result)