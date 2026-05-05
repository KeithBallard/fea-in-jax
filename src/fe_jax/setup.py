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
            V[i * n_cols + j, :] = [x, y, 0.0]

    # Create the faces matrix (defining quadrilaterals)
    F = np.zeros(((n_rows - 1) * (n_cols - 1), 4), dtype=np.int64)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            f = i * (n_cols - 1) + j
            F[f, :] = [
                i * n_cols + j,
                (i + 1) * n_cols + j,
                i * n_cols + j + 1,
                (i + 1) * n_cols + j + 1,
            ]

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
            V[i * n_cols + j, :] = [x, y, 0.0]

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
    vertices: np.ndarray[Any, np.dtype[np.float32 | np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
    number_of_subdivisions: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32 | np.float64]],
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
    cells: np.ndarray[Any, np.dtype[np.uint64]] | np.ndarray[Any, np.dtype[np.int64]],
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
    vertices: np.ndarray[Any, np.dtype[np.float32 | np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.float32 | np.float64]]:
    x_end = np.zeros(
        (cells.shape[0], cells.shape[1], vertices.shape[1]), dtype=vertices.dtype
    )
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            x_end[i, j] = vertices[cells[i, j]]
    return x_end


# @timer()
def mesh_to_jax(
    vertices: np.ndarray[Any, np.dtype[np.float32 | np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jnp.ndarray:
    """
    Given the vertex coordinates and list of connectivity as a list of vertex indices,
    this returns a 3-dimensional arry describing the elements in terms of vertices.

    Returns
    -------
    ```
    For example, in 2D the result would be:
    [ [[e0_v0_x, e0_v0_y],
        [e0_v1_x, e0_v1_y],
        [e0_v2_x, e0_v2_y]],
        ...,
        [[eN_v0_x, eN_v0_y],
        [eN_v1_x, eN_v1_y],
        [eN_v2_x, eN_v2_y]]
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


@partial(jax.jit,static_argnames = "n_vertices")
def mesh_to_sparse_assembly_map(
    n_vertices: int,
    cells: jnp.ndarray,
):
    """
    Builds a compressed sparse row (CSR) matrix that serves as a map between two representations
    for vectors on a patch: 1) assembled format and 2) element-node format.

    Takes advantage of the fact that the assembly map has exactly one entry in each column, so the COO representation has a convenient structure.
    """      
    nse = np.prod(cells.shape)
    coo_data = jnp.ones(nse)
    # Search breaks if the padding is over 50%, so ensure the padding is maxed now.
    coo_rowindices = jnp.searchsorted(
        jnp.arange(n_vertices),
        cells.flatten(),
        method="scan_unrolled",
    )
    coo_colindices = jnp.arange(nse)

    return jsparse.BCOO(
        (
            coo_data,
            jnp.stack((coo_rowindices, coo_colindices)).T,
        ),
        shape=(n_vertices, nse),
    )


@partial(jax.jit, static_argnames=["E", "V", "U"])
def transform_global_to_element_node(
    assembly_map: jsparse.BCOO, v_g: jnp.ndarray, E: int, V: int, U: int
):
    """
    Transforms a vector that represents a global assembled vector into the element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    return jsparse.bcoo_dot_general(
        assembly_map,
        v_g.reshape(v_g.shape[0], v_g.shape[1]),
        dimension_numbers=(((0,), (0,)), ((), ())),
    ).reshape(E, V, U)


@partial(jax.jit, static_argnames=["E"])
def transform_global_unraveled_to_element_node(
    assembly_map: jsparse.BCOO, v_g: jnp.ndarray, E: int
):
    """
    Transforms a vector that represents a global assembled vector that is unraveled into the
    element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    V = assembly_map.shape[0]
    U = v_g.shape[0] // V
    N = assembly_map.shape[1] // E
    return jsparse.bcoo_dot_general(
        assembly_map,
        v_g.reshape(V, U),
        dimension_numbers=(((0,), (0,)), ((), ())),
    ).reshape(E, N, U)


def transform_element_node_to_global_unraveled_nosum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = jsparse.bcoo_dot_general(
        assembly_map,
        jnp.ones(v_en.shape[0] * v_en.shape[1]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return (v_g / n_cell_per_vert[:, jnp.newaxis]).reshape(
        np.prod(v_g.shape)
    )


@jax.jit
def transform_element_node_to_global_unraveled_sum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return v_g.reshape(np.prod(v_g.shape))


@jax.jit
def transform_element_node_to_global_sum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return v_g.reshape(np.prod(v_g.shape) // v_en.shape[2], v_en.shape[2])


@jax.jit
def transform_element_node_to_global_nosum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = jsparse.bcoo_dot_general(
        assembly_map,
        jnp.ones(v_en.shape[0] * v_en.shape[1]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return (v_g / n_cell_per_vert[:, jnp.newaxis]).reshape(np.prod(v_g.shape)//v_en.shape[2],v_en.shape[2])
