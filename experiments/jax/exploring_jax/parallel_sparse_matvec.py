import jax
import jax.experimental
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from helper import *
import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={3}"
print("Devices: ", jax.devices())

N = 6
print(f"Expected vector size: {N / 1e6 * 4} MB")

print(f"initial_mem: {get_current_pid_host_memory()} MB")

v = jnp.ones(shape=(N,), dtype=np.float32)
after_alloc_mem = get_current_pid_host_memory()
print(f"After allocation: {after_alloc_mem} MB")

sharded_v = shard_across_local_devices(v)[0]
sharded_v.block_until_ready()
print("Sharded array layout:")
jax.debug.visualize_array_sharding(sharded_v)
after_shard_mem = get_current_pid_host_memory()
print(f"After sharding: {after_shard_mem} MB")

E = N - 1
cells = np.zeros((E, 2), dtype=np.uint64)
cells[:, 0] = np.arange(E, dtype=np.uint64)
cells[:, 1] = np.arange(E, dtype=np.uint64) + 1
assert cells.shape == (E, 2)
print(cells)

# Create row offset map for the compressed sparse row (CSR) format
csr_row_ind = build_row_ind(N, cells)
assert csr_row_ind.shape == (N + 1,)
csr_col_ind, csr_data = build_col_ind_and_data(csr_row_ind, cells)
print(csr_col_ind)

from scipy.sparse import csr_matrix

csr_matrix_example = csr_matrix((csr_data, csr_col_ind, csr_row_ind), shape=(N, 2 * E))
plt.spy(csr_matrix_example)
plt.title("CSR Matrix Visualization")
# plt.show()


def slice_csr_for_local_sharding(
    csr_row_ind: np.ndarray[Any, np.dtype] | jnp.ndarray,
    csr_col_ind: np.ndarray[Any, np.dtype] | jnp.ndarray,
    csr_data: np.ndarray[Any, np.dtype] | jnp.ndarray,
):
    """
    TODO document
    """

    # If a JAX arary is past in, cast it to a NumPy array. This should be no-copy if the array
    # is already on host memory. However, it will copy the data to host if the array was on
    # GPU memory.
    # TODO verify expected behavior if array is on GPU memory
    csr_row_ind = np.asarray(csr_row_ind)

    print(csr_row_ind.shape)
    csr_row_ind_shard_end = (
        int((csr_row_ind.shape[0] - 1) // jax.local_device_count())
        * jax.local_device_count()
    )
    print(f"csr_row_ind_shard_end: {csr_row_ind_shard_end}")
    rows_per_shard = int(csr_row_ind_shard_end / jax.local_device_count())
    print(f"rows_per_shard: {rows_per_shard}")

    # Note: I need copies of the row pointers in order to modify them
    shard_csr_row_ind_slices = [
        np.array(
            array_slice(
                array=csr_row_ind,
                axis=0,
                start=(rows_per_shard) * i,
                end=rows_per_shard * (i + 1) + 1,
            )
        )
        for i in range(jax.local_device_count())
    ]
    print(f"shard_csr_row_ind_slices: {shard_csr_row_ind_slices}")

    shard_csr_col_ind_slices = [
        array_slice(
            array=csr_col_ind,
            axis=0,
            start=shard_csr_row_ind_slices[i][0],
            end=shard_csr_row_ind_slices[i][-1],
        )
        for i in range(jax.local_device_count())
    ]
    print(f"shard_csr_col_ind_slices: {shard_csr_col_ind_slices}")

    shard_csr_data_slices = [
        array_slice(
            array=csr_data,
            axis=0,
            start=shard_csr_row_ind_slices[i][0],
            end=shard_csr_row_ind_slices[i][-1],
        )
        for i in range(jax.local_device_count())
    ]
    print(f"shard_csr_data_slices: {shard_csr_data_slices}")

    shard_offsets = [s[0] for s in shard_csr_row_ind_slices]
    for i in range(len(shard_csr_row_ind_slices)):
        shard_csr_row_ind_slices[i] -= shard_offsets[i]
    print(f"shard_csr_row_ind_slices: {shard_csr_row_ind_slices}")

    return (
        shard_csr_data_slices,
        shard_csr_col_ind_slices,
        shard_csr_row_ind_slices,
        shard_offsets,
    )


data, col_ind, row_ind, offset = slice_csr_for_local_sharding(
    csr_row_ind, csr_col_ind, csr_data
)

sparse_shards = [
    jax.device_put(jsparse.CSR((d, c, r), shape=(len(row_ind) - 1, 2 * E)), device)
    for d, c, r, o, device in zip(data, col_ind, row_ind, offset, jax.local_devices())
]
print(sparse_shards)
print(dir(sparse_shards[0]))
for sh in sparse_shards:
    print(sh.data.device)


def sparse_mat_mult(sparse_shards, sharded_v):
    result_pieces = [
        jsparse.csr_matvec(m, v.data, transpose=True)
        for m, v in zip(sparse_shards, sharded_v.addressable_shards)
    ]
    #result = jnp.zeros(shape=(sparse_shards[0].shape[1],))
    #for piece in result_pieces:
    #    result += piece
    return result_pieces


print(type(sharded_v.addressable_shards[0]))
print(type(sharded_v.addressable_shards[0].data))
x = sparse_mat_mult(sparse_shards, sharded_v)
print(x)
for i in x:
    print(i.device)
