from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
# import warp as wp
# import newton
import jax.numpy as jnp

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
