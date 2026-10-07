from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
import newton
from .params import pack_fixed_contact_cells

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
        "The Newton/Warp contact searhc path requires optional dependencies "
        "`newton` and `warp`. Use the SciPy cHDTree contact path or install "
        "those packages."
    ) from _NEWTON_WARP_IMPORT_ERROR

# class ContactBackend(Enum):
#     SCIPY_KDTREE = "scipy_kdtree"
#     NEWTON_WARP = "newton_warp"
#     AUTO = "auto"

# class ContactCapacityError(OverflowError):
#     pass
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
    points = np.asarray(points)
    point_diameters = np.asarray(point_diameters)
    point_fiber_ids = np.asarray(point_fiber_ids)
    builder = newton.ModelBuilder()

    for node_id, x in enumerate(np.asarray(points)):
        r=0.5*point_diameters[node_id]
        gap=(contact_search_alpha-1.0)*r

        # cfg=newton.ModelBuilder.ShapeConfig(gap=gap)
        # body=builder.add_body(xform=wp.transform(wp.vec3(*x)))
        cfg=newton.ModelBuilder.ShapeConfig(gap=gap, density=0.0)
        body=builder.add_link(xform=wp.transform(wp.vec3(*x)))
        builder.add_shape_sphere(
            body=body,
            radius=r,
            cfg=cfg,
            label=f"node_{node_id}",
        )

    # Match current self-contact exclusion.
    for i in range(len(points)):
        max_j = min(len(points), i + self_adjacency_block + 1)
        for j in range(i+1, max_j):
            if point_fiber_ids[i] == point_fiber_ids[j]:
                builder.add_shape_collision_filter_pair(i,j)

    # Bypass Newton's default O(N^2) explicit shape contact pair build during finalize,
    # since we use dynamic Sweep-and-Prune (SAP) broadphase at collide time.
    builder._find_shape_contact_pairs = lambda model, **kwargs: None

    model=builder.finalize(skip_all_validations=True)
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

def warp_contact_batch(
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