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
def parallel_swap(arr):
    arr = arr + rank
    # note: this could also use mpi4jax.sendrecv
    if rank == 0:
        # send, then receive
        mpi4jax.send(arr, dest=1, comm=comm)
        other_arr = mpi4jax.recv(arr, source=1, comm=comm)
        print(rank)
    elif rank == 1:
        # receive, then send
        other_arr = mpi4jax.recv(arr, source=0, comm=comm)
        mpi4jax.send(arr, dest=0, comm=comm)
        print(rank)
    else:
        other_arr = jnp.ones((4,1)) * -1

    return other_arr
    
    # Pass the 'comm' object to mpi4jax
    
def parallel_sum(x):
    # Pass the 'comm' object to mpi4jax
    packedObject = mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)
    return packedObject
    
data = jnp.ones((4, 1)) * rank
result = parallel_swap(data)
result = parallel_sum(result)  #okay, this part is working


print(result)