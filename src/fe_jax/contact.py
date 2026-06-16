from dataclasses import dataclass
import jax
from jax import numpy as jnp
import numpy as np
from typing import Callable
import scipy as sp

from fe_jax.basis_quadrature import FiniteElementType

@dataclass
class ContactPreprocessConfig:
    vertices_fiber_ids: np.ndarray
    radius: float
    self_adjacency_block: int
    material_params: jnp.ndarray
    fe_type: FiniteElementType
    constitutive_model: Callable
    contact_pair_generator: Callable

def _validate_point_cloud(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    points = np.asarray(points)
    point_fiber_ids = np.asarray(point_fiber_ids)

    if points.ndim != 2 or points.shape[1] not in (1, 2, 3):
        raise ValueError("points must have shape (N_total, 1), (N_total, 2), or (N_total, 3)")
    if point_fiber_ids.ndim != 1:
        raise ValueError("point_fiber_ids must have shape (N_total,)")
    if point_fiber_ids.shape[0] != points.shape[0]:
        raise ValueError("point_fiber_ids must have the same leading dimension as points")

    return points, point_fiber_ids

def count_initial_contacts(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    radius: float,
    adjacency_block: int
) -> int:
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if adjacency_block <= 0:
        raise ValueError("adjacency_block must be positive")

    N = points.shape[0]

    d = points[:,None,:] - points[None,:,:]
    dist = jnp.linalg.norm(d,axis=-1)
    dist_mask = dist <= radius

    distinct_fiber_mask = point_fiber_ids[:,None] != point_fiber_ids[None,:]
    distinct_upper_mask = jnp.triu(jnp.ones((N,N),dtype=bool), k=1)
    distinct_pair_mask = distinct_fiber_mask & distinct_upper_mask & dist_mask
    n_distinct = jnp.sum(distinct_pair_mask).astype(jnp.int32)

    self_fiber_mask = point_fiber_ids[:,None] == point_fiber_ids[None,:]
    self_upper_mask = jnp.triu(jnp.ones((N,N),dtype=bool), k=1 + adjacency_block)
    self_pair_mask = self_fiber_mask & self_upper_mask & dist_mask
    n_self = jnp.sum(self_pair_mask).astype(jnp.int32)

    return n_distinct + n_self

# def merge_contact_cells(
#     distinct_contacts: jnp.ndarray,
#     n_distinct: jnp.ndarray,
#     self_contacts: jnp.ndarray,
#     n_self: jnp.ndarray,
#     capacity: int,
# ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     """
#     Merge fixed-capacity distinct-contact and self-contact buffers.

#     Parameters
#     ----------
#     distinct_contacts : jnp.ndarray
#         Array of shape (capacity, 2). Rows after n_distinct may be sentinel rows.
#     n_distinct : jnp.ndarray
#         Number of valid rows in distinct_contacts.
#     self_contacts : jnp.ndarray
#         Array of shape (capacity, 2). Rows after n_self may be sentinel rows.
#     n_self : jnp.ndarray
#         Number of valid rows in self_contacts.
#     capacity : int
#         Output capacity.

#     Returns
#     -------
#     contact_cells : jnp.ndarray
#         Array of shape (capacity, 2) containing merged contact pairs.
#         Unused rows are filled with 0.
#     n_contact : jnp.ndarray
#         Number of valid merged contact rows, clipped to capacity.
#     overflowed : jnp.ndarray
#         True if n_distinct + n_self exceeds capacity.
#     """
#     distinct_contacts = jnp.asarray(distinct_contacts)
#     self_contacts = jnp.asarray(self_contacts)
#     n_distinct = jnp.asarray(n_distinct, dtype=jnp.int32)
#     n_self = jnp.asarray(n_self, dtype=jnp.int32)

#     idx = jnp.arange(capacity, dtype=jnp.int32)

#     n_distinct_valid = jnp.minimum(n_distinct, capacity)
#     n_self_valid = jnp.minimum(n_self, capacity)

#     distinct_valid = idx < n_distinct_valid
#     self_valid = idx < n_self_valid

#     all_rows = jnp.concatenate([distinct_contacts, self_contacts], axis=0)
#     all_valid = jnp.concatenate([distinct_valid, self_valid], axis=0)

#     sentinel = jnp.zeros_like(all_rows)
#     all_rows = jnp.where(all_valid[:, None], all_rows, sentinel)

#     # Stable sort: valid rows first, invalid rows last.
#     order = jnp.argsort(~all_valid, stable=True)
#     merged = all_rows[order]

#     n_total = n_distinct + n_self
#     n_contact = jnp.minimum(n_total, capacity)
#     overflowed = n_total > capacity

#     contact_cells = merged[:capacity]

#     return contact_cells, n_contact, overflowed

def distinct_fiber_node2node(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    radius: float
) -> jnp.ndarray:
    """
    Find node-node contact candidates between distinct fibers.

    Parameters
    ----------
    points : array-like, shape (N_total, D)
        Global point coordinates. ``D`` may be 1, 2, or 3.
    point_fiber_ids : array-like, shape (N_total,)
        Fiber id for each point.
    radius : float
        Contact threshold. A node pair is considered in contact if the
        distance between them is <= radius.

    Returns
    -------
    tuple[jnp.ndarray, int, bool]
        ``distinct_contacts`` with shape ``(capacity, 2)``, the number of valid
        distinct contacts, and an overflow flag.
    """
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    if radius <= 0:
        raise ValueError("radius must be positive")

    N = points.shape[0]

    # d = points[:,None,:] - points[None,:,:]
    # dist = jnp.linalg.norm(d,axis=-1)

    # distinct_fiber_mask = point_fiber_ids[:,None] != point_fiber_ids[None,:]
    # upper_mask = jnp.triu(jnp.ones((N,N),dtype=bool), k=1)
    # dist_mask = dist <= radius

    # pair_mask = distinct_fiber_mask & upper_mask & dist_mask

    # i_idx, j_idx = jnp.nonzero(pair_mask)
    # distinct_contacts = jnp.stack([i_idx,j_idx], axis=1)

    candidates = []
    for i in range(N):
        for j in range(i+1, N):
            if point_fiber_ids[i] != point_fiber_ids[j] and np.linalg.norm(points[i]-points[j]) <= radius:
                candidates.append([i,j])
    if len(candidates)==0:
        distinct_contacts = np.zeros((0,2),dtype=np.int32)
    else:
        distinct_contacts = np.array(candidates, dtype=np.int32)

    return distinct_contacts

def self_fiber_node2node(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    radius: float,
    adjacency_block: int
) -> jnp.ndarray:
    """
    Find node-node self-contact candidates within each fiber.

    Parameters
    ----------
    points : array-like, shape (N_total, D)
        Global point coordinates. ``D`` may be 1, 2, or 3.
    point_fiber_ids : array-like, shape (N_total,)
        Fiber id for each point.
    radius : float
        Contact threshold. A node pair is considered in contact if its Euclidean
        distance is <= radius.
    adjacency_block : int
        Minimum index separation to allow self-contact. A value of ``k``
        excludes pairs with ``j - i <= k``.

    Returns
    -------
    tuple[jnp.ndarray, int, bool]
        ``self_contacts`` with shape ``(capacity, 2)``, the number of valid
        self-contact pairs, and an overflow flag.
    """
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if adjacency_block < 0:
        raise ValueError("adjacency_block must be nonnegative")

    # N = points.shape[0]

    # d = points[:,None,:] - points[None,:,:]
    # dist = jnp.linalg.norm(d,axis=-1)

    # same_fiber_mask = point_fiber_ids[:,None] == point_fiber_ids[None,:]
    # upper_mask = jnp.triu(jnp.ones((N,N),dtype=bool), k=1 + adjacency_block)
    # dist_mask = (dist <= radius)

    # pair_mask = same_fiber_mask & upper_mask & dist_mask

    # i_idx,j_idx = jnp.nonzero(pair_mask)
    # self_contacts = jnp.stack([i_idx,j_idx], axis=1)

    candidates = []
    for fiber_id in np.unique(point_fiber_ids):
        global_indeces = np.where(point_fiber_ids == fiber_id)[0]
        fiber = points[global_indeces]
        for i in range(int(fiber.shape[0])):
            for j in range(i+1+adjacency_block,int(fiber.shape[0])):
                if np.linalg.norm(fiber[i]-fiber[j]) <= radius:
                    candidates.append([global_indeces[i],global_indeces[j]])
    if len(candidates)==0:
        self_contacts = np.zeros((0,2),dtype=np.int32)
    else:
        self_contacts = np.array(candidates, dtype=np.int32)


    return self_contacts

def contact_batch(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    adjacency_block: int,
    radius: float,
    distinct_fiber_fn: Callable = distinct_fiber_node2node,
    self_fiber_fn: Callable = self_fiber_node2node,
) -> np.ndarray:
    """
    Find node-node contact candidates from global point and fiber-id arrays.

    This function orchestrates contact detection by calling one detector for
    distinct-fiber pairs and one detector for self-contact. The detector
    functions are injected so alternative contact algorithms can be tested
    without modifying this routine. The returned contact cells are fixed-size
    ``(capacity, 2)`` integer arrays with ``[0, 0]`` sentinel rows for
    unused capacity.

    Parameters
    ----------
    points : array-like, shape (N_total, D)
        Global point coordinates. ``D`` may be 1, 2, or 3.
    point_fiber_ids : array-like, shape (N_total,)
        Fiber id for each point.
    radius : float
        Contact threshold. A node pair is considered in contact if the
        distance between them is <= radius.
    distinct_fiber_fn : Callable
        Function used to detect contact between two different fibers.
        It must accept ``points``, ``point_fiber_ids``, ``capacity``, and
        ``radius``, and return ``(contacts, n_valid, overflowed)``.
    self_fiber_fn : Callable
        Function used to detect self-contact within a single fiber.
        It must accept ``points``, ``point_fiber_ids``, ``capacity``, ``radius``,
        and ``adjacency_block``, and return ``(contacts, n_valid, overflowed)``.

    Returns
    -------
    :p.ndarray
        Fixed-capacity ``(capacity, 2)`` array of contact node pairs. Unused
        rows are filled with 0.
    """
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    if radius <= 0:
        raise ValueError("radius must be positive")

    # distinct_cells = distinct_fiber_fn(
    #     points = points,
    #     point_fiber_ids = point_fiber_ids,
    #     radius = radius
    # )
    # self_cells = self_fiber_fn(
    #     points = points,
    #     point_fiber_ids = point_fiber_ids,
    #     radius = radius,
    #     adjacency_block = adjacency_block
    # )
    kd_tree = sp.spatial.cKDTree(points)
    pairs = np.array(list(kd_tree.query_pairs(r=0.2)))
    if pairs.shape[0] == 0 :
        return np.zeros((0,2),dtype = np.int32)
    distinct_cells = pairs[point_fiber_ids[pairs[:,0]] != point_fiber_ids[pairs[:,1]]]
    self_cells = pairs[point_fiber_ids[pairs[:,0]] == point_fiber_ids[pairs[:,1]]]
    self_cells = self_cells[self_cells[:,1]-self_cells[:,0]>adjacency_block]

    return np.vstack([distinct_cells, self_cells])
