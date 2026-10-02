from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import newton
import jax.numpy as jnp

fabric = read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab")
fabric = refine_fabric(fabric,dX=0.25)
dummy_pair = find_nonzero_length_dummy_contact_pair(points=fabric.points)

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

def sort_contact(mat):
    mat = np.array(mat)
    mat = np.sort(mat, axis = 1)
    idx = np.lexsort((mat[:,1], mat[:,0]))
    return mat[idx]

# KD-tree approach 
def scipy_contact():
    contact_cells = contact_batch(
        points=fabric.points,
        point_fiber_ids=point_fiber_ids,
        adjacency_block=self_adjacency_block,
        point_diameters=point_diameters,
        search2radius_ratio=contact_search_alpha,
    )
    return sort_contact(contact_cells)

def build_warp():
    dummy_pair = find_nonzero_length_dummy_contact_pair(points=fabric.points)
    ctx = build_newton_node_cloud_contact(
        points=fabric.points,
        point_diameters=point_diameters,
        contact_search_alpha=contact_search_alpha,
        self_adjacency_block=self_adjacency_block,
        point_fiber_ids=point_fiber_ids,
        rigid_contact_max=30*len(fabric.points),
    )
    return ctx, dummy_pair

def warp_contact(ctx, dummy_pair):
    contact_cells, active, count, capacity_exhausted = warp_contact_batch(ctx,fabric.points,dummy_pair)
    # return sort_contact(contact_cells[active])
    return contact_cells[active]

def jztree_contact(dummy_pair,k=160):
    rigid_contact_max = 30 * len(fabric.points)
    contact_cells, active, count, capacity_exhausted = jztree_contact_batch(
        points=fabric.points,
        point_fiber_ids=point_fiber_ids,
        adjacency_block=self_adjacency_block,
        point_diameters=point_diameters,
        search2radius_ratio=contact_search_alpha,
        rigid_contact_max=rigid_contact_max,
        dummy_pair=dummy_pair,
        k=k,
    )
    raise_if_jztree_contact_capacity_exhausted(rigid_contact_max, count, capacity_exhausted)
    # return sort_contact(contact_cells[active])
    return contact_cells[active]

def cupyx_contact(dummy_pair):
    rigid_contact_max = 30 * len(fabric.points)
    contact_cells, active, count, capacity_exhausted = cupyx_contact_batch(
        points=fabric.points,
        point_fiber_ids=point_fiber_ids,
        adjacency_block=self_adjacency_block,
        point_diameters=point_diameters,
        search2radius_ratio=contact_search_alpha,
        rigid_contact_max=rigid_contact_max,
        dummy_pair=dummy_pair,
    )
    raise_if_cupyx_kdtree_contact_capacity_exhausted(rigid_contact_max, count, capacity_exhausted)
    # return sort_contact(contact_cells[active])
    return contact_cells[active]