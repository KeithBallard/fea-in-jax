from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .contact_types import ContactCapacityError
from .params import find_nonzero_length_dummy_contact_pair

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.spatial

    CUPYX_AVAILABLE = True
except ImportError:
    cupyx = None
    cp = None
    CUPYX_AVAILABLE = False


def _require_cupyx_kdtree():
    if not CUPYX_AVAILABLE:
        raise ImportError(
            "cupy/cupyx is not installed or available. "
            "Install cupy and/or cupyx to use ContactBackend.CUPYX_KDTREE."
        )


def raise_if_cupyx_kdtree_contact_capacity_exhausted(
    capacity: int,
    count: jnp.ndarray,
    capacity_exhausted: jnp.ndarray,
):
    if bool(capacity_exhausted):
        raise ContactCapacityError(
            f"cupyx_kdtree contact search reached rigid_contact_max={capacity} "
            f"with count={int(count)}. Increase ContactParams.rigid_contact_max."
        )


@partial(jax.jit, static_argnames=("adjacency_block", "rigid_contact_max"))
def _cupyx_mask_and_pack_kernel(
    pairs: jnp.ndarray,
    points_f32: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    point_diameters: jnp.ndarray,
    dummy_pair: jnp.ndarray,
    search2radius_ratio: float,
    adjacency_block: int,
    rigid_contact_max: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    point_radii = 0.5 * point_diameters
    u = pairs[:, 0].astype(jnp.int32)
    v = pairs[:, 1].astype(jnp.int32)

    # 1. Distance condition: dist^2 <= (search2radius_ratio * (r_u + r_v))^2
    diff = points_f32[u] - points_f32[v]
    dist_sq = jnp.sum(diff * diff, axis=-1)
    thresholds = search2radius_ratio * (point_radii[u] + point_radii[v])
    mask_dist = dist_sq <= (thresholds * thresholds)

    # 2. Topology condition: distinct fibers or self-contact past adjacency block
    fiber_u = point_fiber_ids[u]
    fiber_v = point_fiber_ids[v]
    mask_topo = (fiber_u != fiber_v) | (jnp.abs(v - u) > adjacency_block)

    valid = mask_dist & mask_topo

    # 3. Stream compaction to fixed capacity
    count = jnp.sum(valid)
    dest_idx = jnp.cumsum(valid) - 1

    keep = valid & (dest_idx < rigid_contact_max)
    safe_dest = jnp.where(keep, dest_idx, rigid_contact_max)

    dummy_pair = jnp.asarray(dummy_pair, dtype=jnp.int32)
    node0_buf = jnp.full(rigid_contact_max + 1, dummy_pair[0], dtype=jnp.int32)
    node1_buf = jnp.full(rigid_contact_max + 1, dummy_pair[1], dtype=jnp.int32)

    node0 = node0_buf.at[safe_dest].set(u)[:rigid_contact_max]
    node1 = node1_buf.at[safe_dest].set(v)[:rigid_contact_max]

    active = jnp.arange(rigid_contact_max, dtype=jnp.int32) < count
    capacity_exhausted = count >= rigid_contact_max

    contact_cells = jnp.stack([node0, node1], axis=1)
    contact_cells = jnp.where(active[:, None], contact_cells, dummy_pair[None, :])

    return contact_cells, active, count, capacity_exhausted


def cupyx_contact_batch(
    points: jnp.ndarray | np.ndarray,
    point_fiber_ids: jnp.ndarray | np.ndarray,
    adjacency_block: int,
    point_diameters: jnp.ndarray | np.ndarray,
    search2radius_ratio: float,
    rigid_contact_max: int,
    dummy_pair: jnp.ndarray | np.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Find node-node contact candidates using CuPy's GPU KDTree and pack into
    fixed-capacity JAX contact buffers.

    Parameters
    ----------
    points : array-like, shape (N_total, 3)
        Global point coordinates.
    point_fiber_ids : array-like, shape (N_total,)
        Fiber id for each point.
    adjacency_block : int
        Number of adjacent nodes along the same fiber to exclude from self-contact.
    point_diameters : array-like, shape (N_total,)
        Diameter associated with each point.
    search2radius_ratio : float
        Multiplier applied to the pairwise radius sum.
    rigid_contact_max : int
        Fixed capacity for the output contact buffers.
    dummy_pair : array-like, shape (2,), optional
        Sentinel node pair to fill unused capacity slots.

    Returns
    -------
    contact_cells : jnp.ndarray, shape (rigid_contact_max, 2)
        Active contact node pairs, padded with dummy_pair in inactive slots.
    active : jnp.ndarray, shape (rigid_contact_max,)
        Boolean mask indicating valid contact entries.
    count : jnp.ndarray
        Total number of valid contact pairs detected.
    capacity_exhausted : jnp.ndarray
        Boolean flag indicating whether count >= rigid_contact_max.
    """
    _require_cupyx_kdtree()

    if search2radius_ratio <= 0:
        raise ValueError("search2radius_ratio must be positive")

    points_f32 = jnp.asarray(points, dtype=jnp.float32)
    point_fiber_ids = jnp.asarray(point_fiber_ids, dtype=jnp.int32)
    point_diameters = jnp.asarray(point_diameters, dtype=jnp.float32)

    if dummy_pair is None:
        dummy_pair = find_nonzero_length_dummy_contact_pair(points)
    dummy_pair = jnp.asarray(dummy_pair, dtype=jnp.int32)

    point_radii = 0.5 * point_diameters
    query_radius = float(search2radius_ratio * 2.0 * jnp.max(point_radii))

    # 1. Build CuPy KDTree and query candidate pairs on GPU
    pts_cp = cp.asarray(points_f32)
    tree = cupyx.scipy.spatial.KDTree(pts_cp)
    pairs_cp = tree.query_pairs(r=query_radius)

    if pairs_cp.shape[0] == 0:
        contact_cells = jnp.full((rigid_contact_max, 2), dummy_pair[None, :], dtype=jnp.int32)
        active = jnp.zeros(rigid_contact_max, dtype=bool)
        count = jnp.int32(0)
        capacity_exhausted = jnp.bool_(False)
        return contact_cells, active, count, capacity_exhausted

    # 2. Transfer candidate pairs to JAX on GPU (zero-copy via CUDA array interface)
    pairs = jnp.asarray(pairs_cp)

    # 3. Apply exact physical distance and topological masking
    return _cupyx_mask_and_pack_kernel(
        pairs=pairs,
        points_f32=points_f32,
        point_fiber_ids=point_fiber_ids,
        point_diameters=point_diameters,
        dummy_pair=dummy_pair,
        search2radius_ratio=search2radius_ratio,
        adjacency_block=adjacency_block,
        rigid_contact_max=rigid_contact_max,
    )