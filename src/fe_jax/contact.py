from dataclasses import dataclass
import jax
from jax import numpy as jnp
from enum import Enum
import numpy as np
import scipy as sp
from flax import struct

try:
    import newton
    import warp as wp
except ImportError as exc:
    newton = None
    wp = None
    _NEWTON_WARP_IMPORT_ERROR = exc
else:
    _NEWTON_WARP_IMPORT_ERROR = None
from fe_jax.basis_quadrature import FiniteElementType

class ContactBackend(Enum):
    SCIPY_KDTREE = "scipy_kdtree"
    NEWTON_WARP = "newton_warp"
    AUTO = "auto"

class ContactCapacityError(OverflowError):
    pass

def normalize_contact_backend(backend: ContactBackend | str) -> ContactBackend:
    if isinstance(backend,str):
        backend = ContactBackend(backend)
    if not isinstance(backend,ContactBackend):
        raise TypeError("backend must be a ContactBackend or contact backend string")
    return backend

def resolve_contact_backend(backend: ContactBackend | str, auto_uses_newton_warp: bool = True) -> ContactBackend:
    backend = normalize_contact_backend(backend)
    if backend == ContactBackend.AUTO:
        return (
            ContactBackend.NEWTON_WARP
            if NEWTON_WARP_AVAILABLE and auto_uses_newton_warp
            else ContactBackend.SCIPY_KDTREE
        )
    if backend == ContactBackend.NEWTON_WARP:
        _require_newton_warp()
    return backend

NEWTON_WARP_AVAILABLE = newton is not None and wp is not None

def _require_newton_warp():
    if NEWTON_WARP_AVAILABLE:
        return
    raise ImportError(
        "The Newton/Warp contact searhc path requires optional dependencies "
        "`newton` and `warp`. Use the SciPy cHDTree contact path or install "
        "those packages."
    ) from _NEWTON_WARP_IMPORT_ERROR

@dataclass
class ContactParams:
    self_adjacency_block: int
    contact_constitutive_model: jax.tree_util.Partial # such as `elastic_contact_truss_linear`

    D_stiffness_to_E_ratio: float # ratio between stiffness at the diameter distance to the stiffness of the truss elements (D/E) 
    M_stiffness_to_E_ratio: float # ratio between stiffness at M to the stiffness of the truss elements (M/E) 

    M_to_D_ratio: float # M is distance to start ramping up stiffness, so this is the ratio between M and the fiber diameter (M/D)
    C_to_D_ratio: float # C is distance to have a hard stiffness set.
    # It should be C_to_D_ratio<M_to_D_ratio<contact_search_alpha
    contact_search_alpha: float # dimensionless value for search_radius = contact_search_alpha*(radius1+radius2)
    contact_backend: ContactBackend = ContactBackend.AUTO
    rigid_contact_max: int | None = None

@struct.dataclass
class ContactMaterialSpec:
    E_c: float
    area: float
    M_to_D_ratio: float
    C_to_D_ratio: float
    search_alpha: float
    E_min: float

@dataclass
class NewtonContactContext:
    model: object
    state: object
    collision_pipe: object
    contacts: object
    shape_to_node: jnp.ndarray
    shape_to_node_wp: object
    contact_search_jax_callable: object
    contact_capacity: int

CONTACT_E_C_PARAM = 0
CONTACT_AREA_PARAM = 1
CONTACT_RADIUS_0_PARAM = 2
CONTACT_RADIUS_1_PARAM = 3
CONTACT_M_TO_D_PARAM = 4
CONTACT_C_TO_D_PARAM = 5
CONTACT_SEARCH_ALPHA_PARAM = 6
CONTACT_E_MIN_PARAM = 7
CONTACT_ACTIVE_PARAM = 8
CONTACT_MATERIAL_PARAM_COUNT = 9

def find_nonzero_length_dummy_contact_pair(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    for i in range(points.shape[0]):
        for j in range(i+1, points.shape[0]):
            if np.linalg.norm(points[j] - points[i])>0.0:
                return np.array([i,j],dtype=np.int32)
    raise ValueError("fixed-capacity contact requires at least one nonzero-length dummy pair")

def newton_fixed_contact_cells(
    ctx: NewtonContactContext,
    current_points_jax: jnp.ndarray,
    dummy_pair: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    node0, node1, active, count = NewtonContactSearch(ctx, current_points_jax)

    return pack_fixed_contact_cells(
        node0=node0,
        node1=node1,
        active=active,
        count=count,
        capacity=ctx.contact_capacity,
        dummy_pair=dummy_pair,
    )

@jax.jit
def pack_fixed_contact_cells(
    node0: jnp.ndarray,
    node1: jnp.ndarray,
    active: jnp.ndarray,
    count: jnp.ndarray,
    capacity: int,
    dummy_pair: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    capacity_exhausted = count >= capacity

    contact_cells = jnp.stack([node0,node1],axis=1).astype(jnp.int32)
    dummy_pair = jnp.asarray(dummy_pair, dtype=contact_cells.dtype)
    contact_cells = jnp.where(active[:, None], contact_cells, dummy_pair[None, :])

    return contact_cells, active, count, capacity_exhausted

def build_contact_material_params(
    contact_cells: jnp.ndarray,
    point_radii: jnp.ndarray,
    spec: ContactMaterialSpec,
    active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    contact_cells = jnp.asarray(contact_cells, dtype=jnp.int32)
    point_radii = jnp.asarray(point_radii)

    n_contact = contact_cells.shape[0]
    dtype = point_radii.dtype

    if active is None:
        active = jnp.ones((n_contact,), dtype=dtype)
    else:
        active = jnp.asarray(active, dtype=dtype)

    params = jnp.zeros((n_contact, CONTACT_MATERIAL_PARAM_COUNT), dtype=dtype)
    params = params.at[:, CONTACT_E_C_PARAM].set(spec.E_c)
    params = params.at[:, CONTACT_AREA_PARAM].set(spec.area)
    params = params.at[:, CONTACT_RADIUS_0_PARAM].set(point_radii[contact_cells[:, 0]])
    params = params.at[:, CONTACT_RADIUS_1_PARAM].set(point_radii[contact_cells[:, 1]])
    params = params.at[:, CONTACT_M_TO_D_PARAM].set(spec.M_to_D_ratio)
    params = params.at[:, CONTACT_C_TO_D_PARAM].set(spec.C_to_D_ratio)
    params = params.at[:, CONTACT_SEARCH_ALPHA_PARAM].set(spec.search_alpha)
    params = params.at[:, CONTACT_E_MIN_PARAM].set(spec.E_min)
    params = params.at[:, CONTACT_ACTIVE_PARAM].set(active)

    return params

# def fixed_contact_material_params(
#     contact_cells: jnp.ndarray,
#     active: jnp.ndarray,
#     point_radii: jnp.ndarray,
#     contact_E_c: float,
#     contact_A: float,
#     M_to_D_ratio: float,
#     C_to_D_ratio: float,
#     contact_search_alpha: float,
#     contact_E_min: float,
# ) -> jnp.ndarray:
#     active_f = active.astype(point_radii.dtype)
#     r0 = point_radii[contact_cells[:, 0]]
#     r1 = point_radii[contact_cells[:, 1]]

#     return jnp.column_stack([
#         jnp.full_like(active_f, contact_E_c),
#         jnp.where(active, contact_A, 0.0),
#         r0,
#         r1,
#         jnp.full_like(active_f, M_to_D_ratio),
#         jnp.full_like(active_f, C_to_D_ratio),
#         jnp.full_like(active_f, contact_search_alpha),
#         jnp.full_like(active_f,  contact_E_min),
#         active_f,
#     ])


def build_newton_node_cloud_contact(
    points,
    point_diameters,
    contact_search_alpha,
    self_adjacency_block,
    point_fiber_ids,
    *,
    rigid_contact_max=None,
) -> NewtonContactContext:
    _require_newton_warp()
    builder = newton.ModelBuilder()

    for node_id, x in enumerate(np.asarray(points)):
        r=0.5*point_diameters[node_id]
        gap=(contact_search_alpha-1.0)*r

        cfg=newton.ModelBuilder.ShapeConfig(gap=gap)
        body=builder.add_body(xform=wp.transform(wp.vec3(*x)))
        builder.add_shape_sphere(
            body=body,
            radius=r,
            cfg=cfg,
            label=f"node_{node_id}",
        )

    # Match current self-contact exclusion.
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if point_fiber_ids[i] == point_fiber_ids[j] and j-i <= self_adjacency_block:
                builder.add_shape_collision_filter_pair(i,j)

    model=builder.finalize()
    state=model.state()

    # ModelBuilder.finalize() precomputes model.shape_contact_pairs for Newton's
    # default explicit broad phase. For node-cloud contact this can be O(N^2) and
    # consume a large amount of GPU memory. We use broad_phase="sap" below, which
    # builds candidate pairs from AABB overlap at collide time, so the explicit pair
    # array is not needed. Drop the reference here to avoid carrying that memory.
    model.shape_contact_pairs = wp.zeros(0, dtype=wp.vec2i, device=model.device)
    model.shape_contact_pair_count = 0

    n_points = len(points)
    pipe=newton.CollisionPipeline(
        model,
        broad_phase="sap",
        shape_pairs_max=100*n_points,
        rigid_contact_max=rigid_contact_max,
    )
    contacts = pipe.contacts()

    shape_to_node_np = np.arange(len(points), dtype=np.int32)
    shape_to_node_jax = jax.device_put(
        jnp.asarray(shape_to_node_np, dtype=jnp.int32),
        wp.device_to_jax(model.device),
    )
    shape_to_node_wp = wp.array(
        shape_to_node_np,
        dtype=wp.int32,
        device=model.device,
    )

    return NewtonContactContext(
        model=model,
        state=state,
        collision_pipe=pipe,
        contacts=contacts,
        shape_to_node=shape_to_node_jax,
        shape_to_node_wp=shape_to_node_wp,
        contact_search_jax_callable=_build_newton_contact_search_jax_callable(
            state=state,
            collision_pipe=pipe,
            contacts=contacts,
            shape_to_node_wp=shape_to_node_wp,
            contact_capacity=contacts.rigid_contact_max,
        ),
        contact_capacity=contacts.rigid_contact_max,
    )

if NEWTON_WARP_AVAILABLE:
    @wp.kernel
    def _update_node_body_positions_3d(
        points: wp.array2d[wp.float32],
        body_q: wp.array[wp.transform],
    ):
        i=wp.tid()
        x=wp.vec3(points[i,0],points[i,1],points[i,2])
        body_q[i]=wp.transform(x,wp.quat_identity())

    @wp.kernel
    def _copy_newton_contacts_to_fixed_buffers(
        rigid_contact_shape0: wp.array[wp.int32],
        rigid_contact_shape1: wp.array[wp.int32],
        rigid_contact_count: wp.array[wp.int32],
        shape_to_node: wp.array[wp.int32],
        node0_out: wp.array[wp.int32],
        node1_out: wp.array[wp.int32],
        active_out: wp.array[wp.int32],
        count_out: wp.array[wp.int32],
    ):
        i = wp.tid()
        count = rigid_contact_count[0]

        if i == 0:
            count_out[0] = count

        if i < count:
            node0_out[i] = shape_to_node[rigid_contact_shape0[i]]
            node1_out[i] = shape_to_node[rigid_contact_shape1[i]]
            active_out[i] = 1
        else:
            node0_out[i] = 0
            node1_out[i] = 0
            active_out[i] = 0

else:
    _update_node_body_positions_3d = None
    _copy_newton_contacts_to_fixed_buffers = None

def _build_newton_contact_search_jax_callable(
    *,
    state,
    collision_pipe,
    contacts,
    shape_to_node_wp,
    contact_capacity: int,
):
    _require_newton_warp()

    def _newton_contact_search_callable(
        current_points: wp.array2d[wp.float32],
        node0_out: wp.array[wp.int32],
        node1_out: wp.array[wp.int32],
        active_out: wp.array[wp.int32],
        count_out: wp.array[wp.int32],
    ):
        wp.launch(
            _update_node_body_positions_3d,
            dim=current_points.shape[0],
            inputs=[current_points, state.body_q],
        )
        collision_pipe.collide(state, contacts)
        wp.launch(
            _copy_newton_contacts_to_fixed_buffers,
            dim=contact_capacity,
            inputs=[
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_count,
                shape_to_node_wp,
            ],
            outputs=[node0_out, node1_out, active_out, count_out],
        )

    return wp.jax_callable(
        _newton_contact_search_callable,
        num_outputs=4,
        output_dims={
            "node0_out": (contact_capacity,),
            "node1_out": (contact_capacity,),
            "active_out": (contact_capacity,),
            "count_out": (1,),
        },
        has_side_effect=True,
    )


def update_newton_node_body_positions(ctx, current_points):
    _require_newton_warp()
    current_points_jax = jnp.asarray(current_points, dtype=jnp.float32)
    current_points_jax = jax.device_put(
        current_points_jax,
        wp.device_to_jax(ctx.model.device),
    )
    points_wp=wp.from_jax(current_points_jax)
    wp.launch(
        _update_node_body_positions_3d,
        dim=current_points_jax.shape[0],
        inputs=[points_wp, ctx.state.body_q],
        device=ctx.model.device,
    )

def NewtonContactSearch_jitless(ctx: NewtonContactContext, current_points_jax: jnp.ndarray):
    _require_newton_warp()
    update_newton_node_body_positions(ctx=ctx, current_points=current_points_jax)
    ctx.collision_pipe.collide(ctx.state, ctx.contacts)

    shape0=wp.to_jax(ctx.contacts.rigid_contact_shape0)
    shape1=wp.to_jax(ctx.contacts.rigid_contact_shape1)
    count=wp.to_jax(ctx.contacts.rigid_contact_count)[0]

    node0=ctx.shape_to_node[shape0]
    node1=ctx.shape_to_node[shape1]
    active=jnp.arange(ctx.contact_capacity) < count

    return node0, node1, active, count

def NewtonContactSearch(ctx: NewtonContactContext, current_points_jax: jnp.ndarray):
    _require_newton_warp()
    node0, node1, active_i32, count = ctx.contact_search_jax_callable(
        jnp.asarray(current_points_jax, dtype=jnp.float32),
    )
    active = active_i32.astype(jnp.bool)
    count=count[0]

    return node0, node1, active, count

def raise_if_contact_capacity_exhausted(
    ctx: NewtonContactContext,
    count: jnp.ndarray,
    capacity_exhausted: jnp.ndarray,
):
    if bool(capacity_exhausted):
        raise ContactCapacityError(
            f"Newton/Warp contact search reached rigid_contact_max={ctx.contact_capacity} "
            f"with count={int(count)}. Increase ContactParams.rigid_contact_max."
        )

@dataclass
class ContactPreprocessConfig:
    vertices_fiber_ids: np.ndarray
    radius: float
    self_adjacency_block: int
    material_params: jnp.ndarray
    fe_type: FiniteElementType
    constitutive_model: jax.tree_util.Partial
    contact_pair_generator: jax.tree_util.Partial

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
    point_diameters: np.ndarray,
    search2radius_ratio: float,
) -> np.ndarray:
    return scipy_contact_batch(
        points=points,
        point_fiber_ids=point_fiber_ids,
        adjacency_block=adjacency_block,
        point_diameters=point_diameters,
        search2radius_ratio=search2radius_ratio,
    )
