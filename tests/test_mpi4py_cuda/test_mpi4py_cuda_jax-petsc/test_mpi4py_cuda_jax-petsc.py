from mpi4py import MPI
import jax
import jax.numpy as jnp



# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure GPU is available in JAX
if not jax.devices('gpu'):
    raise RuntimeError('No GPU available for JAX!')

