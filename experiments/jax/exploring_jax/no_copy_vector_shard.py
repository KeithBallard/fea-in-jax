import jax
import jax.numpy as jnp
from helper import *
import os

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={jax.local_device_count()}"
)
print('Devices: ', jax.devices())

N = 400_000_000
print(f'Expected vector size: {N / 1e6 * 4} MB')

print(f'initial_mem: {get_current_pid_host_memory()} MB')

v = jnp.ones(shape=(N,), dtype=np.float32)
after_alloc_mem = get_current_pid_host_memory()
print(f'After allocation: {after_alloc_mem} MB')

sharded_v = shard_across_local_devices(v)[0]
sharded_v.block_until_ready()
print('Sharded array layout:')
jax.debug.visualize_array_sharding(sharded_v)
after_shard_mem = get_current_pid_host_memory()
print(f'After sharding: {after_shard_mem} MB')
assert after_shard_mem - after_alloc_mem < N / 1e6 * 4  # make sure the data was not allocated
