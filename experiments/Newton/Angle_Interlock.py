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
    point_list = [
        refine_fiber(fabric_in.get_fiber_points(i,0),n_elements=n_elements,dX=dX)
        for i in range(fabric_in.get_n_bundles())
    ]
    point_list = []
    for b_i in range(fabric_in.get_n_bundles()):
        for f_i in range(fabric_in.get_n_fibers_in_bundle(b_i)):
            point_list.append(
                refine_fiber(
                    points=fabric_in.get_fiber_points(b_i,f_i),
                    n_elements=n_elements,
                    dX=dX
                )
            )
    fiber_offsets = np.concatenate(
        [
            [0],
            np.cumsum([
                point_set.shape[0]
                for point_set in point_list
            ])
        ]
    )
    points = np.vstack(point_list)

    fabric = VTMSFabric(
        name="RefinedFabric",
        material_ids=np.array(
            [
                fabric_in.get_material_id(i)
                for i in range(fabric_in.get_n_bundles())
            ]
        ),
        diameters=np.array(
            [
                fabric_in.get_diameter(i)
                for i in range(fabric_in.get_n_bundles())
            ]
        ),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=fabric_in.bundle_offsets,
    )
    return fabric

fabric = read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab")
fabric = refine_fabric(fabric,dX=0.25)
point_fiber_ids = np.concatenate(
    [
        np.full((fabric.fiber_offsets[i + 1] - fabric.fiber_offsets[i],), i)
        for i in range(fabric.fiber_offsets.shape[0] - 1)
    ]
)
point_diameters=np.concatenate(
    [
        np.full(
            (fabric.fiber_offsets[b_i+1]-fabric.fiber_offsets[b_i],),
            fabric.get_diameter(b_i)
        ) for b_i in range(fabric.get_n_bundles())
    ]
)
self_adjacency_block=10
contact_search_alpha=2.0

# KD-tree approach 
contact_cells = contact_batch(
    points=fabric.points,
    point_fiber_ids=point_fiber_ids,
    adjacency_block=self_adjacency_block,
    point_diameters=point_diameters,
    search2radius_ratio=contact_search_alpha,
)

# Newton-Warp approach 
# ctx = build_newton_node_cloud_contact(
#     points=points,
#     point_diameters=point_diameters,
#     contact_search_alpha=1,
#     self_adjacency_block=20,
#     point_fiber_ids=point_fiber_ids,
#     rigid_contact_max=20*len(fabric,points),
# )

# n0, n1, active, count = NewtonContactSearch(ctx,points)
