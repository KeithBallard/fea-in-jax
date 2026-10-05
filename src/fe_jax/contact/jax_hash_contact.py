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
            f"jaxhash contact search reached rigid_contact_max={capacity} "
            f"with count={int(count)}. Increase ContactParams.rigid_contact_max."
        )


def pad_hash_cells(
    points: jnp.ndarray | np.ndarray,
    query_radius: float,
    pad_size: int = 2,
    cell_capacity_buffer: float = 1.5,
):
    points_np = np.asarray(points)
    domain_min = points_np.min(axis=0) - query_radius * pad_size
    domain_max = points_np.max(axis=0) + query_radius * pad_size

    span = domain_max - domain_min
    N_xyz = (np.floor(span / query_radius) + 1).astype(int)
    Nx, Ny, Nz = int(N_xyz[0]), int(N_xyz[1]), int(N_xyz[2])
    total_cells = int(Nx * Ny * Nz)

    # 1. Compute initial cell counts on CPU in ~1 millisecond:
    floored = np.floor((points_np - domain_min) / query_radius).astype(int)
    hashes = floored[:, 0] + Nx * floored[:, 1] + (Nx * Ny) * floored[:, 2]
    _, counts = np.unique(hashes, return_counts=True)
    initial_max_per_cell = int(counts.max()) if len(counts) > 0 else 0

    # 2. Add headroom for deformation/compression (rounded to multiple of 8):
    C_max = int(np.ceil(initial_max_per_cell * cell_capacity_buffer))
    C_max = max(8, int(((C_max + 7) // 8) * 8))
    return domain_min, Nx, Ny, Nz, total_cells, C_max


FORWARD_CELLS = [
    [ 0,  0, 0],
    [ 1,  0, 0],
    [-1,  1, 0],
    [ 0,  1, 0],
    [ 1,  1, 0],
    [-1, -1, 1],
    [ 0, -1, 1],
    [ 1, -1, 1],
    [-1,  0, 1],
    [ 0,  0, 1],
    [ 1,  0, 1],
    [-1,  1, 1],
    [ 0,  1, 1],
    [ 1,  1, 1]
]


@partial(jax.jit, static_argnames=("Nx", "Ny", "Nz", "total_cells", "C_max", "rigid_contact_max", "adjacency_block"))
def _jaxhash_contact_batch_kernel(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    point_radii: jnp.ndarray,
    domain_min: jnp.ndarray,
    dummy_pair: jnp.ndarray,
    search2radius_ratio: float,
    Nx: int,
    Ny: int,
    Nz: int,
    total_cells: int,
    C_max: int,
    rigid_contact_max: int,
    adjacency_block: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = points.shape[0]
    FORWARD_CELL_OFFSETS = jnp.asarray(
        [fc[0] + fc[1] * Nx + fc[2] * Ny * Nx for fc in FORWARD_CELLS],
        dtype=jnp.int32
    )

    query_radius = search2radius_ratio * 2.0 * jnp.max(point_radii)

    floored_points = jnp.floor((points - domain_min) / query_radius).astype(jnp.int32)
    hashed = floored_points[:, 0] + Nx * floored_points[:, 1] + (Nx * Ny) * floored_points[:, 2]

    sort_idx = jnp.argsort(hashed).astype(jnp.int32)
    sorted_hashed = hashed[sort_idx]

    all_cells = jnp.arange(total_cells, dtype=jnp.int32)
    cell_starts = jnp.searchsorted(sorted_hashed, all_cells, side="left")
    cell_ends = jnp.searchsorted(sorted_hashed, all_cells, side="right")
    grid_capacity_exhausted = jnp.max(cell_ends - cell_starts) > C_max

    sort_idx_padded = jnp.pad(sort_idx, (0, C_max), constant_values=-1)

    def get_particles_for_cell(c):
        start = cell_starts[c]
        count = cell_ends[c] - start
        slice_particles = jax.lax.dynamic_slice(sort_idx_padded, (start,), (C_max,))
        return jnp.where(jnp.arange(C_max, dtype=jnp.int32) < count, slice_particles, -1)

    cell_particles = jax.vmap(get_particles_for_cell)(all_cells)
    is_own_cell = jnp.repeat(FORWARD_CELL_OFFSETS == 0, C_max)

    def check_contacts_for_node(i):
        my_cell = hashed[i]
        target_cells = jnp.clip(my_cell + FORWARD_CELL_OFFSETS, 0, total_cells - 1)
        candidates = cell_particles[target_cells].reshape(-1)

        symmetry_ok = jnp.where(is_own_cell, candidates > i, candidates != i)

        diff = points[i] - points[candidates]
        dist_sq = jnp.sum(diff * diff, axis=-1)
        thresholds = search2radius_ratio * (point_radii[i] + point_radii[candidates])

        is_valid = (
            (candidates >= 0)
            & symmetry_ok
            & (dist_sq <= thresholds * thresholds)
            & ((point_fiber_ids[i] != point_fiber_ids[candidates]) | (jnp.abs(candidates - i) > adjacency_block))
        )
        return candidates, is_valid

    all_candidates, all_is_valid = jax.vmap(check_contacts_for_node)(jnp.arange(N, dtype=jnp.int32))

    # Step 1: Flatten Candidate Pairs
    K = 14 * C_max
    u_flat = jnp.repeat(jnp.arange(N, dtype=jnp.int32), K)
    v_flat = all_candidates.reshape(-1)
    valid_flat = all_is_valid.reshape(-1)

    n0 = jnp.minimum(u_flat, v_flat).astype(jnp.int32)
    n1 = jnp.maximum(u_flat, v_flat).astype(jnp.int32)

    # Step 2: Parallel Stream Compaction with jnp.cumsum
    count = jnp.sum(valid_flat, dtype=jnp.int32)
    dest_idx = (jnp.cumsum(valid_flat) - 1).astype(jnp.int32)

    keep = valid_flat & (dest_idx < rigid_contact_max)
    safe_dest = jnp.where(keep, dest_idx, rigid_contact_max)

    dummy_pair_jnp = jnp.asarray(dummy_pair, dtype=jnp.int32)
    node0_buf = jnp.full(rigid_contact_max + 1, dummy_pair_jnp[0], dtype=jnp.int32)
    node1_buf = jnp.full(rigid_contact_max + 1, dummy_pair_jnp[1], dtype=jnp.int32)

    node0 = node0_buf.at[safe_dest].set(n0)[:rigid_contact_max]
    node1 = node1_buf.at[safe_dest].set(n1)[:rigid_contact_max]

    active = jnp.arange(rigid_contact_max, dtype=jnp.int32) < count
    capacity_exhausted = (count >= rigid_contact_max) | grid_capacity_exhausted

    contact_cells = jnp.stack([node0, node1], axis=1)
    contact_cells = jnp.where(active[:, None], contact_cells, dummy_pair_jnp[None, :])

    return contact_cells, active, count, capacity_exhausted


def jaxhash_contact_batch(
    points: jnp.ndarray | np.ndarray,
    point_fiber_ids: jnp.ndarray | np.ndarray,
    adjacency_block: int,
    point_diameters: jnp.ndarray | np.ndarray,
    search2radius_ratio: float,
    rigid_contact_max: int,
    domain_min: jnp.ndarray | np.ndarray | None = None,
    Nx: int | None = None,
    Ny: int | None = None,
    Nz: int | None = None,
    total_cells: int | None = None,
    C_max: int | None = None,
    dummy_pair: jnp.ndarray | np.ndarray | None = None,
    pad_size: int = 2,
    cell_capacity_buffer: float = 1.5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform fixed-capacity contact search using GPU/CPU pure-JAX spatial hash grid.

    Parameters:
        points: (N, 3) array of particle/vertex coordinates.
        point_fiber_ids: (N,) fiber ID for each point.
        adjacency_block: int, number of neighboring nodes along the same fiber
            to exclude from self-contact.
        point_diameters: (N,) fiber diameter at each node.
        search2radius_ratio: multiplier for contact distance.
        rigid_contact_max: int, maximum contact pair capacity.
        domain_min: (3,) minimum corner of spatial grid. Auto-computed if None.
        Nx, Ny, Nz: Grid cell counts. Auto-computed if None.
        total_cells: Total cell count (Nx * Ny * Nz). Auto-computed if None.
        C_max: Max particles per cell. Auto-computed if None.
        dummy_pair: (2,) int dummy node index pair for padding inactive slots.
        pad_size: Domain padding in units of query radius (default 2).
        cell_capacity_buffer: Multiplier on initial peak cell density (default 1.5).

    Returns:
        contact_cells: (rigid_contact_max, 2) array of contact node pairs.
        active: (rigid_contact_max,) boolean array of active contacts.
        count: scalar integer with total valid contacts found.
        capacity_exhausted: scalar bool indicating buffer or cell overflow.
    """
    if dummy_pair is None:
        dummy_pair = find_nonzero_length_dummy_contact_pair(points)

    points_f32 = jnp.asarray(points, dtype=jnp.float32)
    point_fiber_ids = jnp.asarray(point_fiber_ids, dtype=jnp.int32)
    point_radii = jnp.asarray(0.5 * point_diameters, dtype=jnp.float32)
    dummy_pair = jnp.asarray(dummy_pair, dtype=jnp.int32)

    # Auto-compute grid geometry if not supplied
    if any(param is None for param in (domain_min, Nx, Ny, Nz, total_cells, C_max)):
        query_radius = float(search2radius_ratio * 2.0 * jnp.max(point_radii))
        (
            domain_min,
            Nx,
            Ny,
            Nz,
            total_cells,
            C_max,
        ) = pad_hash_cells(
            points=points_f32,
            query_radius=query_radius,
            pad_size=pad_size,
            cell_capacity_buffer=cell_capacity_buffer,
        )

    domain_min = jnp.asarray(domain_min, dtype=jnp.float32)

    return _jaxhash_contact_batch_kernel(
        points=points_f32,
        point_fiber_ids=point_fiber_ids,
        point_radii=point_radii,
        domain_min=domain_min,
        dummy_pair=dummy_pair,
        search2radius_ratio=float(search2radius_ratio),
        Nx=int(Nx),
        Ny=int(Ny),
        Nz=int(Nz),
        total_cells=int(total_cells),
        C_max=int(C_max),
        rigid_contact_max=int(rigid_contact_max),
        adjacency_block=int(adjacency_block),
    )


# Alias for compatibility with search queries
jaxhash_search = jaxhash_contact_batch