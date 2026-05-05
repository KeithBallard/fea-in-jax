import jax
import jax.numpy as jnp
from helper import *
import os
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Number of processes
if rank == 0:
    print(f"MPI initialized with {size} processes")

# Configure JAX for CPU-only execution
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={size}"
)
if rank == 0:
    print(f"Devices: ", jax.devices())

# global size
N = 400_000_000
if rank == 0:
    print(f'Expected vector size:     {round(N / 1e6 * 4,1)} MB')

# local size
shard_size = N // size
start = rank * shard_size
end = start + shard_size if rank < size - 1 else N  # Adjust last shard if N not divisible by size
local_size = end - start
if rank == 0:
    print(f"Expected shard size:      {round(local_size / 1e6 * 4,1)} MB")

# Create local shard (zero-copy: only allocate this process's portion)
m0 = get_current_pid_host_memory()
local_v = jnp.ones(shape=(local_size,), dtype=np.float32)
m1 = get_current_pid_host_memory()
if rank == 0:
    print(f'Local shard memory:       {round(m1 - m0,1)} MB')

# Test autodifferentiation

# simple function
def func(v):
    return jnp.sum(v ** 2)  # Example function: sum of squares
expected_grad = 2 * local_v  # For v**2, gradient should be 2*v

# Compute local value and gradient for each shard
#m0 = get_current_pid_host_memory()
local_value, local_grad = jax.value_and_grad(func)(local_v)
#m1 = get_current_pid_host_memory()
#if rank == 0:
#    print(f'Local grad+value memory:  {round(m1 - m0,1)} MB')

# Verify gradient locally for each shard
assert jnp.allclose(local_grad, expected_grad, rtol=1e-5), f"Rank {rank}: Local gradient computation failed"
#print(f"Rank {rank}: Local autodiff test passed")

# Aggregate local values to get global value
global_value = comm.allreduce(local_value, op=MPI.SUM)

# Gather local gradients to rank 0
sendbuf = np.array(local_grad)
recvbuf = np.zeros(N, dtype=np.float32) if rank == 0 else None
comm.Gather(sendbuf, recvbuf, root=0)

# Full Vector Computation on Rank 0
if rank == 0:
    grad_sharded_full = recvbuf  # Already contiguous

    # full vector
    m0 = get_current_pid_host_memory()
    v_full = jnp.ones(shape=(N,), dtype=jnp.float32)
    m1 = get_current_pid_host_memory()
    print(f'Full vector memory:       {round(m1 - m0,1)} MB')

    # full grad,value
    #m0 = get_current_pid_host_memory()
    value_full, grad_full = jax.value_and_grad(func)(v_full)
    #m1 = get_current_pid_host_memory()
    #print(f'Full grad+value memory:   {round(m1 - m0,1)} MB')

    # Compare values
    assert jnp.allclose(value_full, global_value, rtol=1e-5), "Values do not match"
    print(f"Full vector value equals sharded global value ({value_full} == {global_value})")

    # Compare gradients
    assert jnp.allclose(grad_full, grad_sharded_full, rtol=1e-5), "Gradients do not match"

# Compute global sum to demonstrate MPI communication
local_sum = jnp.sum(local_v)
global_sum = comm.allreduce(local_sum, op=MPI.SUM)
if rank == 0:
    print(f"Global sum across all ranks: {global_sum} (expected: {N})")

