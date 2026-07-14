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

        # Calculate centroid
        cx = np.mean(coords[:, 0])
        cy = np.mean(coords[:, 1])

        # Sort points counter-clockwise around centroid
        angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
        ccw_order = np.argsort(angles)

        # Find the "bottom-left" node to start the ordering
        # Using lexsort on (x, y) to prioritize smallest y, then smallest x
        y_ccw = coords[ccw_order, 1]
        x_ccw = coords[ccw_order, 0]
        bl_idx = np.lexsort((x_ccw, y_ccw))[0]

        # Shift ccw_order so that the bottom-left node is first
        ccw_order = np.roll(ccw_order, -bl_idx)

        # Map to Basix Quad4 ordering: [0, 1, 3, 2] in CCW perimeter
        order = ccw_order[[0, 1, 3, 2]]

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