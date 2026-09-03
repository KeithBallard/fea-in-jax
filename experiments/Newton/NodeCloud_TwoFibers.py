from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import newton
import jax.numpy as jnp

def refine_fiber(
    points: jnp.ndarray,
    n_elements: int | None,
    dX: float | None= None,
):
    d = np.linalg.norm(np.diff(points,axis=0),axis=1)
    L = np.concatenate([[0.0], np.cumsum(d)])
    if n_elements is None and dX is None:
        raise(RuntimeError("Must provide number of elements in fiber (n_elements) or length of elements (dX), but you provided neither."))
    if n_elements is not None and dX is not None:
        raise(RuntimeError("Must provide either the number of elements in fiber (n_elements) or length of elements (dX), but not both."))
    if n_elements is None:
        n_elements = int(L[-1]/dX + 1)
    spline = sp.interpolate.CubicSpline(L,points)
    y = spline(np.linspace(0,L[-1],n_elements+1))
    return y

def refine_tow(fabric_in,pattern):
    def circle_pack(pattern,outer_diameter):
        centers = build_custom_hex(pattern,1.0)
        temp_diam = 2*(np.linalg.norm(centers,axis=1).max()+0.5)
        diam = outer_diameter/temp_diam
        centers = build_custom_hex(pattern,diam)
        return centers,diam
    refined_N = sum(pattern)

    point_list = []
    diameters = []

    e1 = np.array([1,0,0])
    for i in range(fabric_in.get_n_bundles()):
        p = fabric_in.get_fiber_points(i,0)
        grad = np.gradient(p, axis = 0, edge_order=2)
        grad /= np.linalg.norm(grad, axis =1,keepdims=True)
        rots = [sp.spatial.transform.Rotation.align_vectors([g],[e1])[0] for g in grad]
        mats = np.stack([r.as_matrix() for r in rots])

        c,d =  circle_pack(pattern,fabric_in.get_diameter(i))
        for center in c:
            rotated_center = mats @ np.array([0,center[0],center[1]])
            point_list.append(p + rotated_center)
        diameters.append(d)

    fiber_offsets = np.concatenate([[0],np.cumsum([point_set.shape[0] for point_set in point_list])])
    points = np.vstack(point_list)
    diameters = np.array(diameters)
    fabric = VTMSFabric(
        name="RefinedTowsInFabric",
        material_ids=np.array([fabric_in.get_material_id(i) for i in range(fabric_in.get_n_bundles())]),
        diameters=diameters,
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=refined_N*fabric_in.bundle_offsets,
    )
    return fabric

def refine_fabric(fabric_in, dX, tow_n = 1, n_elements = None):
    point_list = [refine_fiber(fabric_in.get_fiber_points(i,0),n_elements=n_elements,dX=dX) for i in range(fabric_in.get_n_bundles())]
    fiber_offsets = np.concatenate([[0],np.cumsum([point_set.shape[0] for point_set in point_list])])
    points = np.vstack(point_list)

    fabric = VTMSFabric(
        name="RefinedFabric",
        material_ids=np.array([fabric_in.get_material_id(i) for i in range(fabric_in.get_n_bundles())]),
        diameters=np.array([fabric_in.get_diameter(i) for i in range(fabric_in.get_n_bundles())]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=fabric_in.bundle_offsets,
    )
    return fabric

radius = 0.5
length = 10

fibers = [
    refine_fiber(np.array(
        [
            [-length/2, 0, radius/2],
            [ length/2, 0, radius/2]
        ]),n_elements=2),
    refine_fiber(np.array(
        [
            [0, -length/2, -radius/2],
            [0,  length/2, -radius/2]
        ]),n_elements=2)
]
points=np.concatenate(fibers)
point_diameters = [0.5 for i in points]
point_fiber_ids=np.concatenate(
    [
        np.full((fiber.shape[0],),i) for i,fiber in enumerate(fibers)
    ]
)

ctx = build_newton_node_cloud_contact(
    points=points,
    point_diameters=point_diameters,
    contact_search_alpha=1,
    self_adjacency_block=20,
    point_fiber_ids=point_fiber_ids,
)

n0, n1, active, count = NewtonContactSearch(ctx,points)

contact_cells = contact_batch(
    points=points,
    point_fiber_ids=point_fiber_ids,
    adjacency_block=10,
    point_diameters=point_diameters,
    search2radius_ratio=1,
)

# builder = newton.ModelBuilder()
# shape_to_fiber_segment = {}
# for fiber_id,pts in enumerate(fibers):
#     shape_start = builder.shape_count
#     builder.add_rod(
#         positions=pts,
#         radius = radius,
#         label = f'fiber_{fiber_id}',
#         body_frame_origin="start"
#     )
#     for segment_id in range(len(pts) - 1):
#         shape_to_fiber_segment[shape_start + segment_id] = (fiber_id, segment_id)


# model = builder.finalize()
# collision_pipe = newton.CollisionPipeline(
#     model,
#     deterministic=True,
#     requires_grad=True,
# )
# contacts = collision_pipe.contacts()
# state = model.state()
# collision_pipe.collide(state, contacts)

# n_contacts = contacts.rigid_contact_count.numpy()[0]
# print("\nnumber of contacts: ", n_contacts)
# print("shape labels: ", model.shape_label)
# print("shape map: ", shape_to_fiber_segment)

# s0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
# s1 = contacts.rigid_contact_shape1.numpy()[:n_contacts]
# normal = contacts.rigid_contact_normal.numpy()[:n_contacts]
# margin0 = contacts.rigid_contact_margin0.numpy()[:n_contacts]
# margin1 = contacts.rigid_contact_margin1.numpy()[:n_contacts]

# print("shape0:", s0)
# print("shape1:", s1)
# print("normal:", normal)
# print("margin0:", margin0)
# print("margin1:", margin1)

# # Debug-only if requires_grad=True:
# print("p0_world:", contacts.rigid_contact_diff_point0_world.numpy()[:n_contacts])
# print("p1_world:", contacts.rigid_contact_diff_point1_world.numpy()[:n_contacts])
# print("distance:", contacts.rigid_contact_diff_distance.numpy()[:n_contacts])

# for a, b in zip(s0, s1):
#     print(shape_to_fiber_segment[int(a)], "<->", shape_to_fiber_segment[int(b)])
