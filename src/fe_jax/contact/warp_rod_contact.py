from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

try:
    import newton
    import warp as wp
except ImportError as exc:
    newton = None
    wp = None
    _NEWTON_WARP_IMPORT_ERROR = exc
else:
    _NEWTON_WARP_IMPORT_ERROR = None

NEWTON_WARP_AVAILABLE = newton is not None and wp is not None


def _require_newton_warp():
    if NEWTON_WARP_AVAILABLE:
        return
    raise ImportError(
        "The Newton/Warp rod contact path requires optional dependencies "
        "`newton` and `warp`."
    ) from _NEWTON_WARP_IMPORT_ERROR


class ContactCapacityError(OverflowError):
    pass


class RodContactResult(NamedTuple):
    """Container for GPU-native continuous surface-to-surface rod contacts.

    Attributes:
        seg0: Shape index of the first rod segment, shape (rigid_contact_max,).
        seg1: Shape index of the second rod segment, shape (rigid_contact_max,).
        point0: Contact point in local body frame of seg0, shape (rigid_contact_max, 3).
                Note: Local Z coordinate corresponds to distance along the segment from node u.
        point1: Contact point in local body frame of seg1, shape (rigid_contact_max, 3).
        normal: Unit contact normal in world coordinates, shape (rigid_contact_max, 3).
        active: Boolean mask indicating valid contacts, shape (rigid_contact_max,).
        count: Scalar total number of detected contacts.
    """
    seg0: jnp.ndarray
    seg1: jnp.ndarray
    point0: jnp.ndarray
    point1: jnp.ndarray
    normal: jnp.ndarray
    active: jnp.ndarray
    count: jnp.ndarray


@dataclass
class NewtonRodContactContext:
    model: object
    state: object
    collision_pipe: object
    contacts: object
    segment_nodes: jnp.ndarray
    segment_nodes_wp: object
    segment_radii_wp: object
    num_segments: int
    contact_search_jax_callable: object
    contact_capacity: int


if NEWTON_WARP_AVAILABLE:
    @wp.func
    def _quat_between_vectors_robust(from_vec: wp.vec3, to_vec: wp.vec3, eps: float = 1.0e-8) -> wp.quat:
        """Compute rotation quaternion aligning from_vec to to_vec handling 180-deg singularities."""
        d = wp.dot(from_vec, to_vec)
        if d >= 1.0 - eps:
            return wp.quat_identity()
        if d <= -1.0 + eps:
            helper = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(from_vec[0]) >= 0.9:
                helper = wp.vec3(0.0, 1.0, 0.0)
            axis = wp.cross(from_vec, helper)
            axis_len = wp.length(axis)
            if axis_len <= eps:
                axis = wp.cross(from_vec, wp.vec3(0.0, 0.0, 1.0))
                axis_len = wp.length(axis)
            if axis_len <= eps:
                return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
            axis = axis / axis_len
            return wp.quat_from_axis_angle(axis, wp.pi)
        return wp.quat_between_vectors(from_vec, to_vec)

    @wp.kernel
    def _update_rod_segment_body_transforms(
        points: wp.array2d[wp.float32],
        segment_nodes: wp.array2d[wp.int32],
        segment_radii: wp.array[wp.float32],
        body_q: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_transform: wp.array[wp.transform],
    ):
        s = wp.tid()
        u = segment_nodes[s, 0]
        v = segment_nodes[s, 1]

        p0 = wp.vec3(points[u, 0], points[u, 1], points[u, 2])
        p1 = wp.vec3(points[v, 0], points[v, 1], points[v, 2])
        seg_vec = p1 - p0
        seg_len = wp.length(seg_vec)
        half_len = 0.5 * seg_len

        seg_dir = wp.vec3(0.0, 0.0, 1.0)
        if seg_len > 1.0e-7:
            seg_dir = seg_vec / seg_len

        q = _quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), seg_dir)
        body_q[s] = wp.transform(p0, q)
        shape_scale[s] = wp.vec3(segment_radii[s], half_len, 0.0)
        shape_transform[s] = wp.transform(wp.vec3(0.0, 0.0, half_len), wp.quat_identity())

    @wp.kernel
    def _copy_newton_rod_contacts_to_fixed_buffers(
        rigid_contact_shape0: wp.array[wp.int32],
        rigid_contact_shape1: wp.array[wp.int32],
        rigid_contact_point0: wp.array[wp.vec3],
        rigid_contact_point1: wp.array[wp.vec3],
        rigid_contact_normal: wp.array[wp.vec3],
        rigid_contact_count: wp.array[wp.int32],
        seg0_out: wp.array[wp.int32],
        seg1_out: wp.array[wp.int32],
        point0_out: wp.array2d[wp.float32],
        point1_out: wp.array2d[wp.float32],
        normal_out: wp.array2d[wp.float32],
        active_out: wp.array[wp.int32],
        count_out: wp.array[wp.int32],
    ):
        i = wp.tid()
        count = rigid_contact_count[0]
        if i == 0:
            count_out[0] = count

        if i < count:
            seg0_out[i] = rigid_contact_shape0[i]
            seg1_out[i] = rigid_contact_shape1[i]
            p0 = rigid_contact_point0[i]
            p1 = rigid_contact_point1[i]
            n = rigid_contact_normal[i]

            point0_out[i, 0] = p0[0]
            point0_out[i, 1] = p0[1]
            point0_out[i, 2] = p0[2]

            point1_out[i, 0] = p1[0]
            point1_out[i, 1] = p1[1]
            point1_out[i, 2] = p1[2]

            normal_out[i, 0] = n[0]
            normal_out[i, 1] = n[1]
            normal_out[i, 2] = n[2]

            active_out[i] = 1
        else:
            seg0_out[i] = -1
            seg1_out[i] = -1
            point0_out[i, 0] = 0.0
            point0_out[i, 1] = 0.0
            point0_out[i, 2] = 0.0
            point1_out[i, 0] = 0.0
            point1_out[i, 1] = 0.0
            point1_out[i, 2] = 0.0
            normal_out[i, 0] = 0.0
            normal_out[i, 1] = 0.0
            normal_out[i, 2] = 0.0
            active_out[i] = 0
else:
    _quat_between_vectors_robust = None
    _update_rod_segment_body_transforms = None
    _copy_newton_rod_contacts_to_fixed_buffers = None


def _build_newton_rod_contact_search_jax_callable(
    *,
    model,
    state,
    collision_pipe,
    contacts,
    segment_nodes_wp,
    segment_radii_wp,
    num_segments: int,
    contact_capacity: int,
):
    _require_newton_warp()

    def _newton_rod_contact_search_callable(
        current_points: wp.array2d[wp.float32],
        seg0_out: wp.array[wp.int32],
        seg1_out: wp.array[wp.int32],
        point0_out: wp.array2d[wp.float32],
        point1_out: wp.array2d[wp.float32],
        normal_out: wp.array2d[wp.float32],
        active_out: wp.array[wp.int32],
        count_out: wp.array[wp.int32],
    ):
        wp.launch(
            _update_rod_segment_body_transforms,
            dim=num_segments,
            inputs=[
                current_points,
                segment_nodes_wp,
                segment_radii_wp,
                state.body_q,
                model.shape_scale,
                model.shape_transform,
            ],
        )
        collision_pipe.collide(state, contacts)
        wp.launch(
            _copy_newton_rod_contacts_to_fixed_buffers,
            dim=contact_capacity,
            inputs=[
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_count,
            ],
            outputs=[
                seg0_out,
                seg1_out,
                point0_out,
                point1_out,
                normal_out,
                active_out,
                count_out,
            ],
        )

    return wp.jax_callable(
        _newton_rod_contact_search_callable,
        num_outputs=7,
        output_dims={
            "seg0_out": (contact_capacity,),
            "seg1_out": (contact_capacity,),
            "point0_out": (contact_capacity, 3),
            "point1_out": (contact_capacity, 3),
            "normal_out": (contact_capacity, 3),
            "active_out": (contact_capacity,),
            "count_out": (1,),
        },
        has_side_effect=True,
    )


def build_newton_rods_contact(
    fabric,
    self_adjacency_block: int = 1,
    *,
    rigid_contact_max: int | None = None,
    shape_pairs_max: int | None = None,
) -> NewtonRodContactContext:
    """Build a GPU Newton/Warp continuous capsule rod collision model from fabric geometry.

    Args:
        fabric: Fabric object with fiber offsets, bundle offsets, and node coordinates.
        rigid_contact_max: Maximum capacity for detected contacts. Defaults to 30 * len(fabric.points).
        shape_pairs_max: Maximum candidate pairs for broadphase SAP. Defaults to 100 * len(fabric.points).

    Returns:
        NewtonRodContactContext containing the initialized GPU model, state, pipeline,
        and JAX-callable search primitive.
    """
    _require_newton_warp()

    builder = newton.ModelBuilder()
    fiber_id = 0
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    segment_nodes_list = []
    segment_radii_list = []
    seg_global_idx = 0

    for b_i in range(fabric.get_n_bundles()):
        radius = float(0.5 * fabric.get_diameter(b_i))
        for f_i in range(fabric.get_n_fibers_in_bundle(b_i)):
            # points_start = fabric.fiber_offsets[fabric.bundle_offsets[b_i] + f_i]
            # points_end = fabric.fiber_offsets[fabric.bundle_offsets[b_i] + f_i + 1]
            # pts = fabric.points[points_start:points_end]

            # builder.add_rod(
            #     rod=newton.Rod(
            #         points=pts,
            #         radius=radius,
            #     ),
            #     label=f"fiber_{fiber_id}",
            #     body_frame_origin="start",
            # )
            # for seg_i in range(len(pts) - 1):
            #     segment_nodes_list.append([points_start + seg_i, points_start + seg_i + 1])
            points_start = int(fabric.fiber_offsets[fabric.bundle_offsets[b_i] + f_i])
            points_end = int(fabric.fiber_offsets[fabric.bundle_offsets[b_i] + f_i + 1])
            n_segs = points_end - points_start - 1
            fiber_seg_start = seg_global_idx

            for seg_i in range(n_segs):
                u = points_start + seg_i
                v = u + 1
                body_id = builder.add_link()
                builder.add_shape_capsule(
                    body_id,
                    radius=radius,
                    half_height=0.1,  # dynamically updated on GPU per search
                    cfg=cfg,
                )
                segment_nodes_list.append([u, v])
                segment_radii_list.append(radius)
                seg_global_idx += 1
            # fiber_id += 1

            # Exclude self-collisions between adjacent segments along the same fiber
            for s in range(fiber_seg_start, seg_global_idx - 1):
                max_k = min(seg_global_idx, s + self_adjacency_block + 1)
                for k in range(s + 1, max_k):
                    builder.add_shape_collision_filter_pair(s, k)

    num_segments = len(segment_nodes_list)
    segment_nodes_np = np.asarray(segment_nodes_list, dtype=np.int32)
    segment_radii_np = np.asarray(segment_radii_list, dtype=np.float32)

    # Bypass Newton's default O(N^2) explicit shape contact pair build during finalize,
    # since we use dynamic Sweep-and-Prune (SAP) broadphase at collide time.
    builder._find_shape_contact_pairs = lambda model, **kwargs: None

    # model = builder.finalize()
    model = builder.finalize(skip_all_validations=True)
    model.shape_contact_pairs = wp.zeros(0, dtype=wp.vec2i, device=model.device)
    model.shape_contact_pair_count = 0
    state = model.state()

    n_points = len(fabric.points)
    if rigid_contact_max is None:
        rigid_contact_max = 30 * n_points
    if shape_pairs_max is None:
        shape_pairs_max = 100 * n_points

    pipe = newton.CollisionPipeline(
        model,
        broad_phase="sap",
        shape_pairs_max=shape_pairs_max,
        rigid_contact_max=rigid_contact_max,
    )
    contacts = pipe.contacts()

    segment_nodes_jax = jax.device_put(
        jnp.asarray(segment_nodes_np, dtype=jnp.int32),
        wp.device_to_jax(model.device),
    )
    segment_nodes_wp = wp.array2d(
        segment_nodes_np,
        dtype=wp.int32,
        device=model.device,
    )
    segment_radii_wp = wp.array(
        segment_radii_np,
        dtype=wp.float32,
        device=model.device,
    )

    contact_search_callable = _build_newton_rod_contact_search_jax_callable(
        model=model,
        state=state,
        collision_pipe=pipe,
        contacts=contacts,
        segment_nodes_wp=segment_nodes_wp,
        segment_radii_wp=segment_radii_wp,
        num_segments=num_segments,
        contact_capacity=rigid_contact_max,
    )

    return NewtonRodContactContext(
        model=model,
        state=state,
        collision_pipe=pipe,
        contacts=contacts,
        segment_nodes=segment_nodes_jax,
        segment_nodes_wp=segment_nodes_wp,
        segment_radii_wp=segment_radii_wp,
        num_segments=num_segments,
        contact_search_jax_callable=contact_search_callable,
        contact_capacity=rigid_contact_max,
    )


def NewtonRodContactSearch(
    ctx: NewtonRodContactContext,
    current_points_jax: jnp.ndarray,
) -> RodContactResult:
    """Execute GPU-native continuous capsule rod contact search.

    Fully compatible with @jax.jit without host roundtrips.

    Args:
        ctx: NewtonRodContactContext instance.
        current_points_jax: Current (N, 3) node positions array.

    Returns:
        RodContactResult namedtuple containing seg0, seg1, point0, point1, normal, active, count.
    """
    _require_newton_warp()
    seg0, seg1, point0, point1, normal, active_i32, count = ctx.contact_search_jax_callable(
        jnp.asarray(current_points_jax, dtype=jnp.float32),
    )
    return RodContactResult(
        seg0=seg0,
        seg1=seg1,
        point0=point0,
        point1=point1,
        normal=normal,
        active=active_i32.astype(jnp.bool_),
        count=count[0],
    )