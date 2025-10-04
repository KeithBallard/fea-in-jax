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

if rank == 0:
  data = jnp.arange(10, dtype=jnp.float32)
  comm.send(data, dest=1, tag=0)
  print(f'Rank {rank} sent data : {type(data)} {data}')
elif rank == 1:
  data = None
  data = comm.recv(source=0, tag=0)
  print(f'Rank {rank} recv data : {type(data)} {data}')

