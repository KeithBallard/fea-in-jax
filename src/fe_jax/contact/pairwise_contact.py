import jax.numpy as jnp
import numpy as np

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

def distinct_fiber_node2node(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    point_radii: jnp.ndarray | None = None,
    radius: float | None = None,
) -> jnp.ndarray:
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    if radius is not None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        point_radii = np.full(points.shape[0], 0.5 * radius)
    elif point_radii is not None:
        point_radii = np.asarray(point_radii)
    else:
        raise ValueError("Either radius or point_radii must be provided")

    N = points.shape[0]
    candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            if point_fiber_ids[i] != point_fiber_ids[j] and np.linalg.norm(points[i] - points[j]) <= point_radii[i] + point_radii[j]:
                candidates.append([i, j])
    if len(candidates) == 0:
        distinct_contacts = np.zeros((0, 2), dtype=np.int32)
    else:
        distinct_contacts = np.array(candidates, dtype=np.int32)

    return distinct_contacts

def self_fiber_node2node(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    point_radii: jnp.ndarray | None = None,
    adjacency_block: int = 1,
    radius: float | None = None,
) -> jnp.ndarray:
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    if radius is not None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        point_radii = np.full(points.shape[0], 0.5 * radius)
    elif point_radii is not None:
        point_radii = np.asarray(point_radii)
    else:
        raise ValueError("Either radius or point_radii must be provided")

    if adjacency_block < 0:
        raise ValueError("adjacency_block must be nonnegative")

    candidates = []
    for fiber_id in np.unique(point_fiber_ids):
        global_indices = np.where(point_fiber_ids == fiber_id)[0]
        fiber_point_radii = point_radii[global_indices]
        fiber = points[global_indices]
        for i in range(int(fiber.shape[0])):
            for j in range(i + 1 + adjacency_block, int(fiber.shape[0])):
                if np.linalg.norm(fiber[i] - fiber[j]) <= fiber_point_radii[i] + fiber_point_radii[j]:
                    candidates.append([global_indices[i], global_indices[j]])
    if len(candidates) == 0:
        self_contacts = np.zeros((0, 2), dtype=np.int32)
    else:
        self_contacts = np.array(candidates, dtype=np.int32)

    return self_contacts

def pairwise_contact_batch(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    adjacency_block: int,
    point_diameters: np.ndarray,
    search2radius_ratio: float,
) -> np.ndarray:
    if search2radius_ratio <= 0:
        raise ValueError("search2radius_ratio must be positive")

    point_diameters = np.asarray(point_diameters, dtype=np.float64)
    if point_diameters.ndim != 1:
        raise ValueError("point_diameters must have shape (N_total,)")
    if point_diameters.shape[0] != points.shape[0]:
        raise ValueError("point_diameters must have the same leading dimension as points")
    if np.any(point_diameters < 0):
        raise ValueError("point_diameters must be nonnegative")

    point_radii = search2radius_ratio * 0.5 * point_diameters

    self_cells = self_fiber_node2node(
        points = points,
        point_fiber_ids = point_fiber_ids,
        point_radii = point_radii,
        adjacency_block = adjacency_block,
    )
    distinct_cells = distinct_fiber_node2node(
        points = points,
        point_fiber_ids = point_fiber_ids,
        point_radii = point_radii,
    ) 

    return np.vstack([distinct_cells, self_cells])