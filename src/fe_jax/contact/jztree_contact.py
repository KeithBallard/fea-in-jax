from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .contact_types import ContactCapacityError
from .params import find_nonzero_length_dummy_contact_pair

try:
    import jztree as jz
    JZTREE_AVAILABLE = True
except ImportError:
    jz = None
    JZTREE_AVAILABLE = False

def _require_jztree():
    if not JZTREE_AVAILABLE:
        raise ImportError(
            "jztree is not installed or available. "
            "Install jztree to use ContactBackend.JZTREE."
        )

def raise_if_jztree_contact_capacity_exhausted(
    capacity: int,
    count: jnp.ndarray,
    capacity_exhausted: jnp.ndarray,
):
    if bool(capacity_exhausted):
        raise ContactCapacityError(
            f"jztree contact search reached rigid_contact_max={capacity} "
            f"with count={int(count)}. Increase ContactParams.rigid_contact_max."
        )


@partial(jax.jit, static_argnames=("adjacency_block", "rigid_contact_max", "k"))
def _jztree_contact_batch_kernel(
    points_f32: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    point_diameters: jnp.ndarray,
    dummy_pair: jnp.ndarray,
    search2radius_ratio: float,
    adjacency_block: int,
    rigid_contact_max: int,
    k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = points_f32.shape[0]
    point_radii = 0.5 * jnp.asarray(point_diameters, dtype=jnp.float32)
    fiber_ids = jnp.asarray(point_fiber_ids, dtype=jnp.int32)

    # 1. kNN search using jztree on GPU
    rnn, inn = jz.knn.knn(points_f32, k=k)

    # 2. Flatten directed candidates (u -> v) and symmetrize to undirected (min, max)
    u = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[:, None], (N, k)).reshape(-1)
    v = inn.reshape(-1)
    r = rnn.reshape(-1)

    node_min = jnp.minimum(u, v)
    node_max = jnp.maximum(u, v)

    dist_thresh = search2radius_ratio * (point_radii[u] + point_radii[v])
    dist_ok = r <= dist_thresh
    topo_ok = (fiber_ids[u] != fiber_ids[v]) | (jnp.abs(u - v) > adjacency_block)
    valid = (u != v) & dist_ok & topo_ok

    # 3. Encode pairs into 64-bit keys and sort on GPU to deduplicate
    SENTINEL = jnp.int64(0x7FFFFFFFFFFFFFFF)
    keys = (node_min.astype(jnp.int64) << 32) | node_max.astype(jnp.int64)
    keys = jnp.where(valid, keys, SENTINEL)

    sorted_keys = jnp.sort(keys)

    is_dup = jnp.concatenate([jnp.array([False]), sorted_keys[1:] == sorted_keys[:-1]])
    is_valid = (sorted_keys != SENTINEL) & (~is_dup)

    # 4. Stream compaction to fixed capacity
    count = jnp.sum(is_valid)
    dest_idx = jnp.cumsum(is_valid) - 1

    keep = is_valid & (dest_idx < rigid_contact_max)
    safe_dest = jnp.where(keep, dest_idx, rigid_contact_max)

    n0 = (sorted_keys >> 32).astype(jnp.int32)
    n1 = (sorted_keys & 0xFFFFFFFF).astype(jnp.int32)

    dummy_pair = jnp.asarray(dummy_pair, dtype=jnp.int32)
    node0_buf = jnp.full(rigid_contact_max + 1, dummy_pair[0], dtype=jnp.int32)
    node1_buf = jnp.full(rigid_contact_max + 1, dummy_pair[1], dtype=jnp.int32)

    node0 = node0_buf.at[safe_dest].set(n0)[:rigid_contact_max]
    node1 = node1_buf.at[safe_dest].set(n1)[:rigid_contact_max]

    active = jnp.arange(rigid_contact_max, dtype=jnp.int32) < count
    capacity_exhausted = count >= rigid_contact_max

    contact_cells = jnp.stack([node0, node1], axis=1)
    contact_cells = jnp.where(active[:, None], contact_cells, dummy_pair[None, :])

    return contact_cells, active, count, capacity_exhausted


def jztree_contact_batch(
    points: jnp.ndarray | np.ndarray,
    point_fiber_ids: jnp.ndarray | np.ndarray,
    adjacency_block: int,
    point_diameters: jnp.ndarray | np.ndarray,
    search2radius_ratio: float,
    rigid_contact_max: int,
    dummy_pair: jnp.ndarray | np.ndarray | None = None,
    k: int = 128,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform fixed-capacity contact search using GPU-accelerated jztree kNN.

    Parameters:
        points: (N, D) array of particle/vertex coordinates.
        point_fiber_ids: (N,) fiber ID for each point.
        adjacency_block: int, number of neighboring nodes along the same fiber
            to exclude from self-contact.
        point_diameters: (N,) fiber diameter at each node.
        search2radius_ratio: dimensionless multiplier (search_alpha) for contact distance.
        rigid_contact_max: int, maximum number of contact pairs allocated (capacity).
        dummy_pair: (2,) int array of dummy contact node indices for padding.
            Auto-detected if None.
        k: int, number of nearest neighbors per node to search (default: 128).

    Returns:
        contact_cells: (rigid_contact_max, 2) array of contact node pairs.
        active: (rigid_contact_max,) boolean array indicating active contact pairs.
        count: scalar integer with total number of discovered valid contacts.
        capacity_exhausted: scalar bool indicating whether count >= rigid_contact_max.
    """
    _require_jztree()

    if dummy_pair is None:
        dummy_pair = find_nonzero_length_dummy_contact_pair(points)

    points_f32 = jnp.asarray(points, dtype=jnp.float32)
    point_fiber_ids = jnp.asarray(point_fiber_ids, dtype=jnp.int32)
    point_diameters = jnp.asarray(point_diameters, dtype=jnp.float32)
    dummy_pair = jnp.asarray(dummy_pair, dtype=jnp.int32)

    return _jztree_contact_batch_kernel(
        points_f32=points_f32,
        point_fiber_ids=point_fiber_ids,
        point_diameters=point_diameters,
        dummy_pair=dummy_pair,
        search2radius_ratio=float(search2radius_ratio),
        adjacency_block=int(adjacency_block),
        rigid_contact_max=int(rigid_contact_max),
        k=int(k),
    )