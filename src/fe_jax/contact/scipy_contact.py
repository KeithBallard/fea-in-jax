import jax.numpy as jnp
import numpy as np
import scipy as sp

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

def scipy_contact_batch(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    adjacency_block: int,
    point_diameters: np.ndarray,
    search2radius_ratio: float,
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
    point_diameters : array-like, shape (N_total,), optional
        Diameter associated with each point. When provided, contact search uses
        pairwise surface radii instead of the legacy absolute radius.
    search2radius_ratio : float, optional
        Multiplier applied to the pairwise radius sum. Required when
        point_diameters is provided.

    Returns
    -------
    :p.ndarray
        Fixed-capacity ``(capacity, 2)`` array of contact node pairs. Unused
        rows are filled with 0.
    """
    points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
    # if point_diameters is not None:
    if search2radius_ratio <= 0:
        raise ValueError("search2radius_ratio must be positive")

    point_diameters = np.asarray(point_diameters, dtype=np.float64)
    if point_diameters.ndim != 1:
        raise ValueError("point_diameters must have shape (N_total,)")
    if point_diameters.shape[0] != points.shape[0]:
        raise ValueError("point_diameters must have the same leading dimension as points")
    if np.any(point_diameters < 0):
        raise ValueError("point_diameters must be nonnegative")

    point_radii = 0.5 * point_diameters
    query_radius = search2radius_ratio * 2.0 * np.max(point_radii)
    # else:
    #     if search2radius_ratio is not None:
    #         raise ValueError("search2radius_ratio requires point_diameters")
    #     point_radii = None
    #     query_radius = radius



    kd_tree = sp.spatial.cKDTree(points)
    pairs = np.array(list(kd_tree.query_pairs(r=query_radius)),dtype=np.int32)
    if pairs.shape[0] == 0 :
        return np.zeros((0,2),dtype = np.int32)

    pair_distances = np.linalg.norm(
        points[pairs[:, 0], :] - points[pairs[:, 1], :],
        axis=1,
    )
    pair_thresholds = search2radius_ratio * (
        point_radii[pairs[:, 0]] + point_radii[pairs[:, 1]]
    )
    pairs = pairs[pair_distances <= pair_thresholds]
    if pairs.shape[0] == 0:
        return np.zeros((0,2),dtype = np.int32)

    distinct_cells = pairs[point_fiber_ids[pairs[:,0]] != point_fiber_ids[pairs[:,1]]]
    self_cells = pairs[point_fiber_ids[pairs[:,0]] == point_fiber_ids[pairs[:,1]]]
    self_cells = self_cells[self_cells[:,1]-self_cells[:,0]>adjacency_block]

    return np.vstack([distinct_cells, self_cells])

def contact_batch(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    adjacency_block: int,
    point_diameters: np.ndarray | None = None,
    search2radius_ratio: float = 1.0,
    radius: float | None = None,
    distinct_fiber_fn = None,
    self_fiber_fn = None,
) -> np.ndarray:
    if radius is not None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        points, point_fiber_ids = _validate_point_cloud(points, point_fiber_ids)
        if distinct_fiber_fn is not None and self_fiber_fn is not None:
            distinct_contacts = distinct_fiber_fn(
                points=points,
                point_fiber_ids=point_fiber_ids,
                radius=radius,
            )
            self_contacts = self_fiber_fn(
                points=points,
                point_fiber_ids=point_fiber_ids,
                radius=radius,
                adjacency_block=adjacency_block,
            )
            return jnp.concatenate([distinct_contacts, self_contacts], axis=0)
        else:
            point_diameters = np.full(points.shape[0], radius)
            search2radius_ratio = 1.0

    return scipy_contact_batch(
        points=points,
        point_fiber_ids=point_fiber_ids,
        adjacency_block=adjacency_block,
        point_diameters=point_diameters,
        search2radius_ratio=search2radius_ratio,
    )