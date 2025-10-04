# Explores sparse matrices and sharding in JAX

import jax
import jax.experimental
import jax.numpy as jnp
from jax.experimental import mesh_utils, sparse
from jax import random, vmap
from helper import *
import psutil
import os
import scipy.sparse

os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={10}'
print(jax.devices())

############################################
# Create a random sparse matrix and dense vector

n_rows = 1000
n_non_zeros = 1000

def sample_two(key, num_items):
  return random.choice(key , num_items, (2,), replace=False)

key_array = random.split(random.PRNGKey(0), num=n_non_zeros)
indices = vmap(sample_two, in_axes=(0, None))(key_array, n_rows)
data = random.uniform(random.PRNGKey(1), (n_non_zeros,))

A = sparse.COO((data, indices[:, 0], indices[:, 1]), shape=(n_rows, n_rows))
print(A)

b = random.uniform(random.PRNGKey(0), (n_rows,))

A.block_until_ready()
b.block_until_ready()
#timeit(sparse.coo_matvec, {'mat': A, 'v': b}, generated_kwargs={}, n_calls=10)

############################################
# Test sharding

D = jax.local_device_count()
devices = mesh_utils.create_device_mesh((D,))
mesh = jax.sharding.Mesh(devices, axis_names=("ax"))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("ax"))

print(f"Number of logical devices: {len(devices)}")
print(f"Shape of device array    : {devices.shape}")
print(f"\nMesh     : {mesh}")
print(f"Sharding : {sharding}\n\n")

sharded_A = jax.device_put(A, sharding)
sharded_b = jax.device_put(b, sharding)
sharded_A.block_until_ready()
sharded_b.block_until_ready()

#timeit(sparse.coo_matvec, {'mat': sharded_A, 'v': sharded_b}, generated_kwargs={}, n_calls=10)


############################################
# Test sharding with CSR matrices

n_rows = 1000
n_non_zeros = 1000

key_array = random.split(random.PRNGKey(0), num=n_non_zeros)
indices = vmap(sample_two, in_axes=(0, None))(key_array, n_rows)
data = random.uniform(random.PRNGKey(1), (n_non_zeros,))

A_scipy = scipy.sparse.coo_matrix((np.array(data), (np.array(indices[:,0]), np.array(indices[:,1]))), shape=(n_rows, n_rows))
A_scipy = A_scipy.tocsr()
A_csr = sparse.CSR((A_scipy.data, A_scipy.indices, A_scipy.indptr), shape=A_scipy.shape)
b = random.uniform(random.PRNGKey(0), (n_rows,))

sharded_A_csr = jax.device_put(A_csr, sharding)
sharded_A_csr.block_until_ready()
sharded_b = jax.device_put(b, sharding)
sharded_b.block_until_ready()

timeit(sparse.csr_matvec, {'mat': sharded_A_csr, 'v': sharded_b}, generated_kwargs={}, n_calls=10)
