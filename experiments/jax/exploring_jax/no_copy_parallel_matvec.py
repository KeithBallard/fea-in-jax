import jax
import jax.numpy as jnp
from helper import *
import os

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={jax.local_device_count()}"
)
print(jax.devices())

N = 60_000
print(N / 1e6 * 4)

mem1 = get_current_pid_host_memory()
print(mem1)

v = jnp.ones(shape=(N,), dtype=np.float32)
print('after alloc', get_current_pid_host_memory())
mem2 = get_current_pid_host_memory()
print(mem2)

sharded_v = shard_across_local_devices(v)[0]
sharded_v.block_until_ready()
print(sharded_v.addressable_shards)
mem3 = get_current_pid_host_memory()
print(mem3)
assert mem3 - mem2 < N / 1e6 * 4  # make sure the data was not allocated

A = np.ones(shape=(N, N), dtype=jnp.float32)
mem4 = get_current_pid_host_memory()
print(mem4)

sharded_A = shard_across_local_devices(A)[0]
sharded_A.block_until_ready()
print(sharded_A.addressable_shards)
mem5 = get_current_pid_host_memory()
print(mem5)
assert mem5 - mem4 < 2.0  # accounts for small overhead of the sharded array

timeit(jnp.matmul, {"a": A, "b": v}, generated_kwargs={}, n_calls=10)
timeit(jnp.matmul, {"a": sharded_A, "b": sharded_v}, generated_kwargs={}, n_calls=10)
