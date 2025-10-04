from .basis_quadrature import *
from .utils import *

import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import jax.experimental.sparse as jsparse
from functools import partial

from numba import njit

from typing import Any

import igl


@njit
def uniform_quad_grid(n_rows: int, n_cols: int, bbox):
    """
    Creates a uniform grid of quadrilaters with a specified extent for both x and y.

    Parameters
    ----------
    n_rows  : int, number of rows of vertices
    n_cols  : int, number of columns of vertices
    bbox     : array with shape (D, 2)

    Returns
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)
    """

    # Create the grid coordinates
    x_start = bbox[0, 0]
    x_stop = bbox[0, 1]
    y_start = bbox[1, 0]
    y_stop = bbox[1, 1]

    # Create the vertices matrix
    V = np.zeros((n_rows * n_cols, 3), dtype=np.float64)
    for i in range(n_rows):
        x = i / float(n_rows - 1) * (x_stop - x_start) + x_start
        for j in range(n_cols):
            y = j / float(n_cols - 1) * (y_stop - y_start) + y_start
            V[i * n_cols + j, :] = [x, y, 0.]

    # Create the faces matrix (defining quadrilaterals)
    F = np.zeros(((n_rows - 1) * (n_cols - 1), 4), dtype=np.int64)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            f = i *  (n_cols - 1) +  j
            F[f, :] = [i * n_cols + j, (i + 1) * n_cols + j, i * n_cols + j + 1, (i + 1) * n_cols + j + 1]

    return (V, F)



@njit
def uniform_tri_grid(n_rows: int, n_cols: int):
    """
    Creates a uniform grid of triangles with an extent for both x and y of [0, 1].

    Parameters
    ----------
    n_rows  : int, number of rows of vertices
    n_cols  : int, number of columns of vertices

    Returns
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)
    """

    # Create the grid coordinates
    x_start = 0.0
    x_stop = 1.0
    y_start = 0.0
    y_stop = 1.0

    # Create the vertices matrix
    V = np.zeros((n_rows * n_cols, 3), dtype=np.float64)
    for i in range(n_rows):
        x = i / float(n_rows - 1) * (x_stop - x_start) + x_start
        for j in range(n_cols):
            y = j / float(n_cols - 1) * (y_stop - y_start) + y_start
            V[i * n_cols + j, :] = [x, y, 0.]

    # Create the faces matrix (defining triangles)
    F = np.zeros((2 * (n_rows - 1) * (n_cols - 1), 3), dtype=np.int64)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            f = i * 2 * (n_cols - 1) + 2 * j
            F[f, :] = [i * n_cols + j, (i + 1) * n_cols + j, i * n_cols + j + 1]
            f += 1
            F[f, :] = [
                (i + 1) * n_cols + j,
                (i + 1) * n_cols + j + 1,
                i * n_cols + j + 1,
            ]

    return (V, F)


def refine_tri_mesh(
    vertices: np.ndarray[Any, np.dtype[np.float32|np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
    number_of_subdivisions: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32|np.float64]],
    np.ndarray[Any, np.dtype[np.uint64]],
]:
    """
    Given a triangle mesh, this subdivides each triangle uniformly to create a more refined mesh
    without changing the coarse features of the mesh.

    Parameters
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)

    Returns
    -------
    refined_vertices    : dense 2d-array with shape (# verts, 3)
    refined_cells       : dense 2d-array with shape (# elements, 3)

    """
    return igl.upsample(V=vertices, F=cells, number_of_subdivs=number_of_subdivisions)


def find_tri_mesh_boundary_verts(
    cells: np.ndarray[Any, np.dtype[np.uint64]] | np.ndarray[Any, np.dtype[np.int64]]
) -> np.ndarray[Any, np.dtype[np.uint64]] | np.ndarray[Any, np.dtype[np.int64]]:
    """
    Given a triangle mesh, this finds the vertices along the boundary of the mesh.

    Parameters
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)

    Returns
    -------
    boundary_verts    : dense 1d-array with shape (# boundary verts,)
    """

    boundary_line_segments = igl.boundary_facets(cells)[0]
    return np.unique(boundary_line_segments)


@njit
def mesh_to_jax_helper(
    vertices: np.ndarray[Any, np.dtype[np.float32|np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.float32|np.float64]]:
    x_n = np.zeros((cells.shape[0], cells.shape[1], 2), dtype=vertices.dtype)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            # Note: [0:2] is added to ignore z-coordinate if one exists.
            x_n[i, j] = vertices[cells[i, j]][0:2]
    return x_n


# @timer()
def mesh_to_jax(
    vertices: np.ndarray[Any, np.dtype[np.float32|np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jnp.ndarray:
    """
    Given the vertex coordinates and list of triangles as a triplet of vertex indices,
    this returns a 3-dimensional arry describing the triangles in terms of vertices.

    Returns
    -------
    ```
    [ [[t0_v0_x, t0_v0_y],
        [t0_v1_x, t0_v1_y],
        [t0_v2_x, t0_v2_y]],
        ...,
        [[tN_v0_x, tN_v0_y],
        [tN_v1_x, tN_v1_y],
        [tN_v2_x, tN_v2_y]]
    ]
    ```
    """
    return jnp.array(mesh_to_jax_helper(vertices, cells))


@njit
def get_n_cells_per_vert_helper(
    vertices: np.ndarray[Any, np.dtype[np.float32]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    n_cells_per_vert = np.zeros((vertices.shape[0],), dtype=np.uint64)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            n_cells_per_vert[cells[i, j]] += 1
    return n_cells_per_vert


# @timer()
def get_n_cells_per_vert(
    vertices: np.ndarray[Any, np.dtype[np.floating]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jnp.ndarray:
    """
    Returns an array that describes the number of cells connected to each vertex.
    """
    return jnp.array(get_n_cells_per_vert_helper(vertices, cells))


@njit
def build_row_ind(
    n_vertices: int, cells: np.ndarray[Any, np.dtype[np.uint64]]
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """
    Creates row offset map for the compressed sparse row (CSR) format.
    """
    csr_row_ind = np.zeros((n_vertices + 1,), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            csr_row_ind[np.uint64(cells[i, j]) + one] += 1
    return np.cumsum(csr_row_ind)


@njit
def build_col_ind_and_data(
    csr_row_ind: np.ndarray[Any, np.dtype[np.uint64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> tuple[np.ndarray[Any, np.dtype[np.uint64]], np.ndarray[Any, np.dtype[np.float32|np.float64]]]:
    """
    Create column offset map and data for the compressed sparse row (CSR) format.
    """
    n_verts_per_cell = cells.shape[1]
    curr_col_offset = np.zeros((csr_row_ind.shape[0],), dtype=np.uint64)
    csr_col_ind = np.zeros((csr_row_ind[-1],), dtype=np.uint64)
    csr_data = np.zeros((csr_row_ind[-1],))
    for cell_index, cell in enumerate(cells):
        for i, vert_index in enumerate(cell):
            index = csr_row_ind[vert_index] + curr_col_offset[vert_index]
            col = cell_index * n_verts_per_cell + i
            csr_col_ind[index] = col
            csr_data[index] = 1.0
            curr_col_offset[vert_index] += 1
    return (csr_col_ind, csr_data)


# @timer()
def mesh_to_sparse_assembly_map(
    n_vertices: int,
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jsparse.BCSR:
    """
    Builds a compressed sparse row (CSR) matrix that serves as a map between two representations
    for vectors: 1) globally assembled format and 2) batched element-node format.

    NOTE: A map can only be created for a single type of cell, so different cell types need
    to be organized into separate batches.

    Parameters
    ----------
    vertices    : dense 2d-array with shape (# verts, N_x)
    cells       : dense 2d-array with shape (# elements, V)

    Returns
    -------
    assembly_map : sparse 3d-array with shape (# batches, # verts, # elements * V)
    """

    # Create row offset map for the compressed sparse row (CSR) format
    csr_row_ind = build_row_ind(n_vertices, cells)
    csr_col_ind, csr_data = build_col_ind_and_data(csr_row_ind, cells)

    # Make a batch of 1
    csr_row_ind = csr_row_ind.reshape(1, csr_row_ind.shape[0])
    csr_col_ind = csr_col_ind.reshape(1, csr_col_ind.shape[0])
    csr_data = csr_data.reshape(1, csr_data.shape[0])

    return jsparse.BCSR(
        (csr_data, csr_col_ind, csr_row_ind),
        shape=(1, n_vertices, cells.shape[0] * cells.shape[1]),
        indices_sorted=True,
        unique_indices=True,
    )


@partial(jax.jit,static_argnames = ["E","V","U"])
def transform_global_to_element_node(
    assembly_map: jsparse.BCSR, v_g: jnp.ndarray, E: int, V: int, U: int
):
    """
    Transforms a vector that represents a global assembled vector into the element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    return jsparse.bcsr_dot_general(
        assembly_map,
        v_g.reshape(1, v_g.shape[0], v_g.shape[1]),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
    ).reshape(E, V, U)


@partial(jax.jit, static_argnames=["E"])
def transform_global_unraveled_to_element_node(
    assembly_map: jsparse.BCSR, v_g: jnp.ndarray, E: int
):
    """
    Transforms a vector that represents a global assembled vector that is unraveled into the
    element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    assert assembly_map.shape[0] == 1
    V = assembly_map.shape[1]
    U = v_g.shape[0] // V
    N = assembly_map.shape[2] // E
    return jsparse.bcsr_dot_general(
        assembly_map,
        v_g.reshape(1, V, U),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
    ).reshape(E, N, U)


def transform_element_node_to_global_unraveled_nosum(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = (
        assembly_map.indptr[0, 1:] - assembly_map.indptr[0, :-1]
    )
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return (v_g / n_cell_per_vert[jnp.newaxis, :, jnp.newaxis]).reshape(
        np.prod(v_g.shape)
    )


@jax.jit
def transform_element_node_to_global_unraveled_sum(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return v_g.reshape(np.prod(v_g.shape))

@jax.jit
def transform_element_node_to_global_sum(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return v_g.reshape(np.prod(v_g.shape)//v_en.shape[2],v_en.shape[2])

@jax.jit
def transform_element_node_to_global_nosum(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = (
        assembly_map.indptr[0, 1:] - assembly_map.indptr[0, :-1]
    )
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return (v_g / n_cell_per_vert[jnp.newaxis, :, jnp.newaxis]).reshape(np.prod(v_g.shape)//v_en.shape[2],v_en.shape[2])
    