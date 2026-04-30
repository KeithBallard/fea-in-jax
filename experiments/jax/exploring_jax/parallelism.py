import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from helper import *

import psutil
import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={psutil.cpu_count()}' # Use max CPU devices
print(jax.devices())

# Set up plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('10 Vector-Vector Inner Products')

with poll_cpu() as cpu_poll:

    #################################################
    # Set up

    A = jax.random.uniform(jax.random.key(10), shape=(100_000, 100_000), dtype=jnp.float32)
    v = jax.random.uniform(jax.random.key(11), shape=(100_000,), dtype=jnp.float32)
    A.block_until_ready()
    v.block_until_ready()

    cpu_poll.stop()
    cpu_poll.get_plt_fig(ax1)
    ax1.set_title('Create Random Arrays')

    #################################################
    # Case 1: No JIT and No Sharding

    cpu_poll.restart()
    timeit(jnp.matmul, {'a': A, 'b': v}, generated_kwargs={}, n_calls=10)
    cpu_poll.stop()
    cpu_poll.get_plt_fig(ax2)
    ax2.set_title('No JIT - No Sharding')

    #################################################
    # Case 2: JIT and No Sharding

    matmul_jit = jax.jit(jnp.matmul)
    cpu_poll.restart()
    timeit(matmul_jit, {'a': A, 'b': v}, generated_kwargs={}, n_calls=10)
    cpu_poll.stop()
    cpu_poll.get_plt_fig(ax3)
    ax3.set_title('JIT - No Sharding')


    #################################################
    # Case 3: JIT and Sharding

    from jax.sharding import PositionalSharding

    cpu_poll.restart()
    
    D = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((D,))
    mesh = jax.sharding.Mesh(devices, axis_names=("ax"))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("ax"))

    print(f"Number of logical devices: {len(devices)}")
    print(f"Shape of device array    : {devices.shape}")
    print(f"\nMesh     : {mesh}")
    print(f"Sharding : {sharding}\n\n")

    D = jax.local_device_count()
    cpu_poll.mark_event('Begin device_put')
    sharded_A = jax.device_put(A, sharding)
    sharded_v = jax.device_put(v, sharding)
    del A, v
    sharded_A.block_until_ready()
    sharded_v.block_until_ready()
    print(f'sharded_v shape: {sharded_v.shape}')
    cpu_poll.mark_event('End device_put')

    timeit(jnp.matmul, {'a': sharded_A, 'b': sharded_v}, generated_kwargs={}, n_calls=10)

    cpu_poll.stop()
    cpu_poll.get_plt_fig(ax4)
    ax4.set_title('Sharding')

plt.tight_layout()
plt.show()