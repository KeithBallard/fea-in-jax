from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .contact_types import ContactCapacityError
from .params import find_nonzero_length_dummy_contact_pair

def raise_if_jaxhash_contact_capacity_exhausted(
    capacity: int,
    count: jnp.ndarray,
    capacity_exhausted: jnp.ndarray,
):
    if bool(capacity_exhausted):
        raise ContactCapacityError(
            f"jaxjash contact search reached rigid_contact_max={capacity} "
            f"with count={int(count)}. Increase ContactParams.rigid_contact_max."
        )

def jaxhash_contact_batch(
    points: jnp.ndarray | np.ndarray,
    point_fiber_ids: jnp.ndarray | np.ndarray,
    adjacency_block: int,
    point_diameters: jnp.ndarray | np.ndarray,
    search2radius_ratio: float,
    rigid_contact_max: int,
    dummy_pair: jnp.ndarray | np.ndarray | None = None,
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