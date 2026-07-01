import sys
import os
import zipfile

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from fe_jax import *

import meshio
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import chain
import os
from pathlib import Path

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)


def get_mesh(mesh_name: str):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "meshes", mesh_name
    )


def get_fabric(fabric_name: str):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "fabrics",
        fabric_name,
        f"{fabric_name}.fab",
    )


def get_output(filename: str):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/output", exist_ok=True)
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", filename)


def reorder_cell_basix(mesh, point_ids):
    """
    Reorders TRI3 or QUAD4 node indices to match Basix ordering convention.

    For QUAD4: lexicographic ordering on unit square [0,1]^2
    For TRI3 : lexicographic ordering based on (x, y)

    Parameters
    ----------
    mesh : pyvista mesh
    point_ids : list or array of node indices in the cell

    Returns
    -------
    np.ndarray
        Reordered point indices to match Basix convention.
    """
    tol = 1e-7
    coords = mesh.points[point_ids][:, :2]  # only x-y needed

    if len(point_ids) == 3:  # triangle
        # leave as is.
        order = np.array([0,1,2])
    elif len(point_ids) == 4:  # quad
        # Sort by (y, x) → Basix lexicographic order: x increases fastest
        x = coords[:, 0]
        y = coords[:, 1]

        # Rank y values to split bottom/top robustly
        y_sorted = np.sort(y)
        # Threshold between the 2nd and 3rd smallest y (middle split)
        split_val = 0.5 * (y_sorted[1] + y_sorted[2])

        # Handle near-ties: if the gap is tiny, nudge the split
        if abs(y_sorted[2] - y_sorted[1]) < 1e-7:
            split_val += tol

        bottom_mask = y <= split_val
        top_mask = ~bottom_mask

        bottom_idx = np.where(bottom_mask)[0]
        top_idx = np.where(top_mask)[0]

        # Within each half, sort by x ascending
        bl_br = bottom_idx[np.argsort(x[bottom_idx])]
        tl_tr = top_idx[np.argsort(x[top_idx])]

        # Handle near-ties: if the gap is tiny, nudge the split
        if abs(x[top_idx[0]] - x[top_idx[1]]) < 1e-7:
            # vertical RHS of bottom 
            if x[top_idx[0]]> x[bottom_idx[0]]:
                tl_tr = top_idx[np.argsort(y[top_idx])]
                tl_tr = tl_tr[::-1]
            # vertical LHS of bottom
            elif x[top_idx[0]]< x[bottom_idx[0]]:
                tl_tr = top_idx[np.argsort(y[top_idx])]
        # top is float        
        elif abs(y[top_idx[0]] - y[top_idx[1]]) < 1e-7 and abs(x[bottom_idx[0]] - x[bottom_idx[1]]) < 1e-7:
            if x[top_idx[0]]> x[bottom_idx[0]]:
                bl_br = bl_br[::-1]

        order = np.concatenate([bl_br, tl_tr])
        # order = np.array([bl_br[0], tl_tr[0],bl_br[1], tl_tr[1]])

    else:
        raise ValueError(f"Unsupported element with {len(point_ids)} nodes")

    return np.array(point_ids)[order]


def find_print_cell_idx(mesh,print_cell_ID,matrix_tri_id,matrix_quad_id,fiber_tri_id,fiber_quad_id):
    '''
    Parameters
    ----------
    mesh : pyvista mesh
    point_ids : Pyvista cell ID 

    Returns
    -------
    the index of the cell ID
    '''
    tri_quad = mesh.celltypes[print_cell_ID]
    materials_ID = mesh.cell_data['materials'][print_cell_ID]
    matrix_ID = np.max(mesh.cell_data['materials'])

    if tri_quad == 5:
        if materials_ID == matrix_ID:
            fib_matrix_shape = 0
            print_cell = np.where(matrix_tri_id==print_cell_ID)[0][0]
        else:
            fib_matrix_shape = 2
            print_cell = np.where(fiber_tri_id==print_cell_ID)[0][0]
    elif tri_quad == 9:
        if materials_ID == matrix_ID:
            fib_matrix_shape = 1
            print_cell = np.where(matrix_quad_id==print_cell_ID)[0][0]
        else:
            fib_matrix_shape = 3
            print_cell = np.where(fiber_quad_id==print_cell_ID)[0][0]
    return print_cell,fib_matrix_shape


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=folder_path)
                zipf.write(abs_path, arcname=rel_path)