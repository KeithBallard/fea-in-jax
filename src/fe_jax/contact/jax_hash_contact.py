import ctypes
from functools import partial
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np

from .contact_types import ContactCapacityError
from .params import find_nonzero_length_dummy_contact_pair


def get_cuda_hardware_info() -> Dict[str, Any]:
    """Query CUDA device attributes directly via libcuda.so with fallback for non-CUDA/CPU.

    Returns:
        Dict containing sm_count, max_threads_per_sm, max_threads_per_block,
        hardware_concurrency, cuda_cores, and discovery source.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so")
        cuda.cuInit(0)
        dev = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(dev), 0)

        sm_count = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, dev)  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

        threads_per_sm = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_sm), 39, dev)  # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR

        threads_per_block = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_block), 1, dev)  # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK

        sms = sm_count.value
        t_sm = threads_per_sm.value
        concurrency = sms * t_sm
        cuda_cores_per_sm = 64  # Volta (V100) / Ampere (A100) standard baseline
        return {
            "sm_count": sms,
            "max_threads_per_sm": t_sm,
            "max_threads_per_block": threads_per_block.value,
            "hardware_concurrency": concurrency,
            "cuda_cores": sms * cuda_cores_per_sm,
            "source": "libcuda.so",
        }
    except Exception:
        devices = jax.devices()
        dev_kind = str(devices[0].device_kind).lower() if devices else "cpu"
        if "v100" in dev_kind:
            sms, t_sm, cores_per_sm = 80, 2048, 64
        elif "a100" in dev_kind:
            sms, t_sm, cores_per_sm = 108, 2048, 64
        elif "h100" in dev_kind:
            sms, t_sm, cores_per_sm = 132, 2048, 128
        else:
            sms, t_sm, cores_per_sm = 80, 2048, 64
        concurrency = sms * t_sm
        return {
            "sm_count": sms,
            "max_threads_per_sm": t_sm,
            "max_threads_per_block": 1024,
            "hardware_concurrency": concurrency,
            "cuda_cores": sms * cores_per_sm,
            "source": f"fallback_lookup ({dev_kind})",
        }


def compute_dynamic_batch_params(
    N: int, C_max: int, safe_vram_fraction: float = 0.8
) -> Tuple[int, int, Dict[str, Any]]:
    """Determine optimal node batch size n_batch and number of batches num_batches based on available VRAM and hardware concurrency.

    Parameters:
        N: Total number of query nodes.
        C_max: Maximum node capacity per grid cell.
        safe_vram_fraction: Target fraction of remaining unallocated VRAM to utilize.

    Returns:
        n_batch: Number of query nodes per batch chunk.
        num_batches: Total number of scanned batch iterations.
        hw_info: CUDA hardware info dict.
    """
    hw_info = get_cuda_hardware_info()
    min_batch_size = max(1024, hw_info["hardware_concurrency"])  # Full wave saturation on all SMs

    # Estimate transient candidate evaluation bytes per node
    # Each node checks 14 * C_max candidate pairs
    candidates_per_node = 14 * C_max
    bytes_per_candidate = 4 * 6  # ~24 bytes for u_flat, v_flat, valid_flat, n0, n1, dist_sq, thresholds
    bytes_per_node = max(1, candidates_per_node * bytes_per_candidate)

    # Query device memory stats from JAX
    devices = jax.devices()
    stats = devices[0].memory_stats() if hasattr(devices[0], "memory_stats") else {}
    bytes_limit = stats.get("bytes_limit", 16 * 1024**3)
    bytes_in_use = stats.get("bytes_in_use", 0)
    avail_vram = max(512 * 1024**2, (bytes_limit - bytes_in_use) * safe_vram_fraction)

    # Reserve 2 GiB for persistent spatial grid, node arrays, and output compaction buffers
    vram_for_candidate_batches = max(256 * 1024**2, avail_vram - 2 * 1024**3)

    max_batch_nodes = int(vram_for_candidate_batches // bytes_per_node)
    max_batch_nodes = max(min_batch_size, max_batch_nodes)

    # Quantize n_batch to a multiple of min_batch_size
    n_batch = int(((max_batch_nodes + min_batch_size - 1) // min_batch_size) * min_batch_size)

    if n_batch >= N:
        n_batch = N
        num_batches = 1
    else:
        num_batches = int((N + n_batch - 1) // n_batch)

    return n_batch, num_batches, hw_info


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


@partial(
    jax.jit,
    static_argnames=(
        "Nx",
        "Ny",
        "Nz",
        "total_cells",
        "C_max",
        "adjacency_block",
        "n_batch",
        "num_batches",
    ),
)
def _jaxhash_contact_count_scanned_kernel(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
    point_radii: jnp.ndarray,
    domain_min: jnp.ndarray,
    search2radius_ratio: float,
    Nx: int,
    Ny: int,
    Nz: int,
    total_cells: int,
    C_max: int,
    adjacency_block: int,
    n_batch: int,
    num_batches: int,
) -> jnp.ndarray:
    N = points.shape[0]
    FORWARD_CELL_OFFSETS = jnp.asarray(
        [fc[0] + fc[1] * Nx + fc[2] * Ny * Nx for fc in FORWARD_CELLS],
        dtype=jnp.int32,
    )

    query_radius = search2radius_ratio * 2.0 * jnp.max(point_radii)

    floored_points = jnp.floor((points - domain_min) / query_radius).astype(jnp.int32)
    floored_points = jnp.clip(
        floored_points,
        0,
        jnp.asarray([Nx - 1, Ny - 1, Nz - 1], dtype=jnp.int32),
    )
    hashed = floored_points[:, 0] + Nx * floored_points[:, 1] + (Nx * Ny) * floored_points[:, 2]

    sort_idx = jnp.argsort(hashed).astype(jnp.int32)
    sorted_hashed = hashed[sort_idx]

    all_cells = jnp.arange(total_cells, dtype=jnp.int32)
    cell_starts = jnp.searchsorted(sorted_hashed, all_cells, side="left")
    cell_ends = jnp.searchsorted(sorted_hashed, all_cells, side="right")

    sort_idx_padded = jnp.pad(sort_idx, (0, C_max), constant_values=-1)

    def get_particles_for_cell(c):
        start = cell_starts[c]
        count = cell_ends[c] - start
        slice_particles = jax.lax.dynamic_slice(sort_idx_padded, (start,), (C_max,))
        return jnp.where(jnp.arange(C_max, dtype=jnp.int32) < count, slice_particles, -1)

    cell_particles = jax.vmap(get_particles_for_cell)(all_cells)
    is_own_cell = jnp.repeat(FORWARD_CELL_OFFSETS == 0, C_max)

    def scan_step(total_count_acc, batch_i):
        batch_start = batch_i * n_batch
        node_indices = batch_start + jnp.arange(n_batch, dtype=jnp.int32)
        valid_query_mask = node_indices < N
        safe_node_indices = jnp.where(valid_query_mask, node_indices, 0)

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
            return is_valid

        is_valid_batch = jax.vmap(check_contacts_for_node)(safe_node_indices)
        valid_flat = is_valid_batch.reshape(-1) & jnp.repeat(valid_query_mask, 14 * C_max)
        batch_count = jnp.sum(valid_flat, dtype=jnp.int32)
        return total_count_acc + batch_count, None

    final_count, _ = jax.lax.scan(
        scan_step, jnp.int32(0), jnp.arange(num_batches, dtype=jnp.int32)
    )
    return final_count


@partial(
    jax.jit,
    static_argnames=(
        "Nx",
        "Ny",
        "Nz",
        "total_cells",
        "C_max",
        "rigid_contact_max",
        "adjacency_block",
        "n_batch",
        "num_batches",
    ),
)
def _jaxhash_contact_scanned_kernel(
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
    n_batch: int,
    num_batches: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = points.shape[0]
    FORWARD_CELL_OFFSETS = jnp.asarray(
        [fc[0] + fc[1] * Nx + fc[2] * Ny * Nx for fc in FORWARD_CELLS],
        dtype=jnp.int32,
    )

    query_radius = search2radius_ratio * 2.0 * jnp.max(point_radii)

    floored_points = jnp.floor((points - domain_min) / query_radius).astype(jnp.int32)
    floored_points = jnp.clip(
        floored_points,
        0,
        jnp.asarray([Nx - 1, Ny - 1, Nz - 1], dtype=jnp.int32),
    )
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

    dummy_pair_jnp = jnp.asarray(dummy_pair, dtype=jnp.int32)
    node0_buf = jnp.full(rigid_contact_max + 1, dummy_pair_jnp[0], dtype=jnp.int32)
    node1_buf = jnp.full(rigid_contact_max + 1, dummy_pair_jnp[1], dtype=jnp.int32)

    def scan_step(carry, batch_i):
        node0_acc, node1_acc, count_acc, overflow_acc = carry

        batch_start = batch_i * n_batch
        node_indices = batch_start + jnp.arange(n_batch, dtype=jnp.int32)
        valid_query_mask = node_indices < N
        safe_node_indices = jnp.where(valid_query_mask, node_indices, 0)

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

        candidates_batch, is_valid_batch = jax.vmap(check_contacts_for_node)(safe_node_indices)

        K = 14 * C_max
        u_flat = jnp.repeat(safe_node_indices, K)
        v_flat = candidates_batch.reshape(-1)
        query_valid = jnp.repeat(valid_query_mask, K)
        valid_flat = is_valid_batch.reshape(-1) & query_valid

        n0 = jnp.minimum(u_flat, v_flat).astype(jnp.int32)
        n1 = jnp.maximum(u_flat, v_flat).astype(jnp.int32)

        batch_count = jnp.sum(valid_flat, dtype=jnp.int32)
        dest_idx = count_acc + (jnp.cumsum(valid_flat) - 1).astype(jnp.int32)

        keep = valid_flat & (dest_idx < rigid_contact_max)
        safe_dest = jnp.where(keep, dest_idx, rigid_contact_max)

        node0_acc = node0_acc.at[safe_dest].set(n0)
        node1_acc = node1_acc.at[safe_dest].set(n1)

        new_count = count_acc + batch_count
        new_overflow = overflow_acc | (new_count >= rigid_contact_max)

        return (node0_acc, node1_acc, new_count, new_overflow), None

    init_carry = (node0_buf, node1_buf, jnp.int32(0), grid_capacity_exhausted)
    (final_node0, final_node1, total_count, capacity_exhausted), _ = jax.lax.scan(
        scan_step, init_carry, jnp.arange(num_batches, dtype=jnp.int32)
    )

    node0 = final_node0[:rigid_contact_max]
    node1 = final_node1[:rigid_contact_max]

    active = jnp.arange(rigid_contact_max, dtype=jnp.int32) < total_count
    capacity_exhausted = capacity_exhausted | (total_count >= rigid_contact_max)

    contact_cells = jnp.stack([node0, node1], axis=1)
    contact_cells = jnp.where(active[:, None], contact_cells, dummy_pair_jnp[None, :])

    return contact_cells, active, total_count, capacity_exhausted

def estimate_jaxhash_rigid_contact_max(
    points: jnp.ndarray | np.ndarray,
    point_fiber_ids: jnp.ndarray | np.ndarray,
    point_diameters: jnp.ndarray | np.ndarray,
    search2radius_ratio: float,
    adjacency_block: int,
    grid_params: Tuple[jnp.ndarray, int, int, int, int, int],
    rigid_contact_factor: float = 1.5,
    n_batch: int | None = None,
) -> int:
    """Precompute rigid_contact_max once from the reference configuration."""
    domain_min, Nx, Ny, Nz, total_cells, C_max = grid_params
    points_f32 = jnp.asarray(points, dtype=jnp.float64)
    point_fiber_ids = jnp.asarray(point_fiber_ids, dtype=jnp.int32)
    point_radii = jnp.asarray(0.5 * point_diameters, dtype=jnp.float64)
    N = points_f32.shape[0]

    if n_batch is None:
        n_batch, num_batches, _ = compute_dynamic_batch_params(N, int(C_max))
    else:
        num_batches = int((N + n_batch - 1) // n_batch)

    prelim_count = int(
        _jaxhash_contact_count_scanned_kernel(
            points=points_f32,
            point_fiber_ids=point_fiber_ids,
            point_radii=point_radii,
            domain_min=jnp.asarray(domain_min, dtype=jnp.float64),
            search2radius_ratio=float(search2radius_ratio),
            Nx=int(Nx),
            Ny=int(Ny),
            Nz=int(Nz),
            total_cells=int(total_cells),
            C_max=int(C_max),
            adjacency_block=int(adjacency_block),
            n_batch=int(n_batch),
            num_batches=int(num_batches),
        )
    )
    return max(16, int(np.ceil(prelim_count * rigid_contact_factor)))

def jaxhash_contact_batch(
    points: jnp.ndarray | np.ndarray,
    point_fiber_ids: jnp.ndarray | np.ndarray,
    adjacency_block: int,
    point_diameters: jnp.ndarray | np.ndarray,
    search2radius_ratio: float,
    rigid_contact_max: int | None = None,
    rigid_contact_factor: float = 1.5,
    domain_min: jnp.ndarray | np.ndarray | None = None,
    Nx: int | None = None,
    Ny: int | None = None,
    Nz: int | None = None,
    total_cells: int | None = None,
    C_max: int | None = None,
    dummy_pair: jnp.ndarray | np.ndarray | None = None,
    pad_size: int = 2,
    cell_capacity_buffer: float = 1.5,
    n_batch: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform fixed-capacity contact search using GPU/CPU pure-JAX spatial hash grid with dynamic nodal batching.

    Parameters:
        points: (N, 3) array of particle/vertex coordinates.
        point_fiber_ids: (N,) fiber ID for each point.
        adjacency_block: int, number of neighboring nodes along the same fiber
            to exclude from self-contact.
        point_diameters: (N,) fiber diameter at each node.
        search2radius_ratio: multiplier for contact distance.
        rigid_contact_max: int | None, maximum contact pair capacity. If None,
            computed dynamically from a preliminary count pass scaled by rigid_contact_factor.
        rigid_contact_factor: float, safety multiplier used when rigid_contact_max is None (default 1.2).
        domain_min: (3,) minimum corner of spatial grid. Auto-computed if None.
        Nx, Ny, Nz: Grid cell counts. Auto-computed if None.
        total_cells: Total cell count (Nx * Ny * Nz). Auto-computed if None.
        C_max: Max particles per cell. Auto-computed if None.
        dummy_pair: (2,) int dummy node index pair for padding inactive slots.
        pad_size: Domain padding in units of query radius (default 2).
        cell_capacity_buffer: Multiplier on initial peak cell density (default 1.5).
        n_batch: int | None, explicit node batch chunk size. If None, dynamically
            computed based on hardware thread concurrency and available VRAM.

    Returns:
        contact_cells: (rigid_contact_max, 2) array of contact node pairs.
        active: (rigid_contact_max,) boolean array of active contacts.
        count: scalar integer with total valid contacts found.
        capacity_exhausted: scalar bool indicating buffer or cell overflow.
    """
    if dummy_pair is None:
        dummy_pair = find_nonzero_length_dummy_contact_pair(points)

    points_f32 = jnp.asarray(points, dtype=jnp.float64)
    point_fiber_ids = jnp.asarray(point_fiber_ids, dtype=jnp.int32)
    point_radii = jnp.asarray(0.5 * point_diameters, dtype=jnp.float64)
    dummy_pair = jnp.asarray(dummy_pair, dtype=jnp.int32)
    N = points_f32.shape[0]

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

    domain_min = jnp.asarray(domain_min, dtype=jnp.float64)

    # Compute dynamic batch parameters if n_batch is not explicitly set
    if n_batch is None:
        n_batch, num_batches, _ = compute_dynamic_batch_params(N, int(C_max))
    else:
        num_batches = int((N + n_batch - 1) // n_batch)

    jax.debug.print("n_batch = {nb}\n num_batches = {num_b}",nb=n_batch,num_b=num_batches)
    if rigid_contact_max is None:
        prelim_count = int(
            _jaxhash_contact_count_scanned_kernel(
                points=points_f32,
                point_fiber_ids=point_fiber_ids,
                point_radii=point_radii,
                domain_min=domain_min,
                search2radius_ratio=float(search2radius_ratio),
                Nx=int(Nx),
                Ny=int(Ny),
                Nz=int(Nz),
                total_cells=int(total_cells),
                C_max=int(C_max),
                adjacency_block=int(adjacency_block),
                n_batch=int(n_batch),
                num_batches=int(num_batches),
            )
        )
        rigid_contact_max = max(16, int(np.ceil(prelim_count * rigid_contact_factor)))
        jax.debug.print("rigid_contact_max = {rgm}",rgm=rigid_contact_max)

    return _jaxhash_contact_scanned_kernel(
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
        n_batch=int(n_batch),
        num_batches=int(num_batches),
    )


# Alias for compatibility with search queries
jaxhash_search = jaxhash_contact_batch