"""
Utilities for meshes produced by the IGFEM code (PIGFEM, run through run_code.sh).

An IGFEM output mesh is the integration (sub-cell) mesh of the cut background triangulation:
TRI3/QUAD4 cells, VTK_POLYGON cells where a background triangle touches two inclusions, a
'materials' cell field in which the matrix has the largest material ID and each fiber has its own
ID, and enrichment nodes stored as ordinary points. These functions sort its cells into element
batches by material and cell shape, reorder their nodes to the element conventions, and warn
about cells that the IGFEM VTK writer collapsed to (nearly) zero area by rounding coordinates to 6
significant digits.
"""

import warnings

import numpy as np

from fe_jax.basis_quadrature import (
    CellType,
    ElementFamily,
    LagrangeVariant,
    QuadratureType,
    FiniteElementType,
    PolygonElementType,
    eval_basis_and_derivatives,
    get_quadrature,
)

VTK_TRIANGLE, VTK_POLYGON, VTK_QUAD = 5, 7, 9


def cell_shape(celltype, n_nodes):
    """Element shape of a VTK cell: "tri", "quad" or "polygon<n_nodes>"."""
    if celltype == VTK_TRIANGLE:
        return "tri"
    if celltype == VTK_QUAD:
        return "quad"
    if celltype == VTK_POLYGON:
        return f"polygon{n_nodes}"
    raise ValueError(f"Unsupported VTK cell type {celltype}")


def igfem_element_type(shape):
    """
    Finite element type for a shape returned by cell_shape. Polygons use PIGFEM's isoparametric
    Wachspress element and quadrature (4-point triangle rule on n sub-triangles), which is what
    IGFEM integrates them with; a fan triangulation into TRI3 does not reproduce it.
    """
    if shape == "tri":
        cell_type, quadrature_degree = CellType.triangle, 1
    elif shape == "quad":
        cell_type, quadrature_degree = CellType.quadrilateral, 2
    elif shape.startswith("polygon"):
        return PolygonElementType(n_vertices=int(shape[len("polygon"):]))
    else:
        raise ValueError(f"Unknown cell shape {shape!r}")
    return FiniteElementType(
        cell_type=cell_type,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=quadrature_degree,
    )


def reorder_cell_basix(mesh, point_ids, celltype):
    """
    Reorders cell node indices to the convention of the element that integrates the cell.

    For QUAD4  : lexicographic ordering on unit square [0,1]^2 (Basix)
    For TRI3   : left as is
    For polygon: counter-clockwise perimeter order (PolygonElementType reference n-gon)

    Parameters
    ----------
    mesh : pyvista mesh
    point_ids : list or array of node indices in the cell
    celltype : int
        VTK cell type: VTK_TRIANGLE, VTK_QUAD or VTK_POLYGON (integrated as a Wachspress element
        whatever its node count). Any other type raises a ValueError, since skipping the cell
        would leave a hole in the domain.

    Returns
    -------
    np.ndarray
        Reordered point indices to match the element convention.
    """
    tol = 1e-7
    coords = mesh.points[point_ids][:, :2]  # only x-y needed

    if celltype == VTK_POLYGON:
        # VTK lists polygon nodes around the perimeter; reverse clockwise cells (the reference n-gon
        # and its quadrature are symmetric, so the start node does not matter)
        x, y = coords[:, 0], coords[:, 1]
        signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        order = np.arange(len(point_ids)) if signed_area > 0.0 else np.arange(len(point_ids))[::-1]
    elif celltype == VTK_TRIANGLE:
        # leave as is.
        order = np.array([0,1,2])
    elif celltype == VTK_QUAD:
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
        raise ValueError(f"Unsupported VTK cell type {celltype} (only TRI3/QUAD4/POLYGON have an element)")

    return np.array(point_ids)[order]


# 2x2 Gauss points on the unit square, matching the quad quadrature_degree=2 used in the tests
_GAUSS_2X2_1D = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))


def cell_shape_ratio(coords):
    """
    Measures how close a TRI3, QUAD4 or polygon cell is to collapsing, independently of its size.

    Element stiffness scales like (edge length)^2 / det(J), so a cell whose det(J) is tiny
    compared with its squared edge length makes the global system singular or indefinite,
    whereas a cell that is merely small does not. The ratio is evaluated at the quadrature
    points the solver actually uses (centroid for TRI3, 2x2 Gauss for QUAD4, the Wachspress
    element quadrature for polygons with more than 4 nodes).

    Parameters
    ----------
    coords : ndarray[float, (N, 2)]
        Node coordinates in VTK order (perimeter order for quads and polygons).

    Returns
    -------
    float
        min det(J) / (longest edge)^2 over the quadrature points, signed relative to the cell's
        overall orientation: ~1 for a square, ~0.87 for an equilateral triangle, 0 for a
        collapsed cell, negative if det(J) changes sign inside the cell (inverted / bowtie).
    """
    coords = np.asarray(coords, dtype=np.float64)
    longest_edge_sq = np.max(np.sum((np.roll(coords, -1, axis=0) - coords) ** 2, axis=1))

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    if len(coords) == 3:
        dets = np.array([cross(coords[1] - coords[0], coords[2] - coords[0])])
    elif len(coords) == 4:
        x0, x1, x2, x3 = coords
        xi, eta = np.meshgrid(_GAUSS_2X2_1D, _GAUSS_2X2_1D, indexing="ij")
        xi, eta = xi.ravel()[:, None], eta.ravel()[:, None]
        dx_dxi = (1.0 - eta) * (x1 - x0) + eta * (x2 - x3)
        dx_deta = (1.0 - xi) * (x3 - x0) + xi * (x2 - x1)
        dets = cross(dx_dxi, dx_deta)
    else:
        fe_type = PolygonElementType(n_vertices=len(coords))
        _, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type, get_quadrature(fe_type)[0])
        dets = np.linalg.det(np.einsum("nd,qnp->qdp", coords, dphi_dxi_qnp))

    orientation = 1.0 if np.sum(dets) >= 0.0 else -1.0
    return float(np.min(orientation * dets) / longest_edge_sq)


def sort_cells_by_material(mesh, fiber_group=None, degenerate_tol=1e-8):
    """
    Splits the cells of an IGFEM mesh into material groups and, within each group, by cell shape
    (see cell_shape), reorders their nodes to the element convention (see reorder_cell_basix),
    and warns about nearly collapsed cells. Only the shapes present in the mesh appear, so the element batches
    can be built by looping over the result, with igfem_element_type(shape) as the element type.

    IGFEM writes node coordinates with 6 significant digits, so integration sub-cells that are
    only ~1e-7 across can be rounded into collinear (zero-area) cells. Such a cell has an
    effectively infinite, randomly signed stiffness instead of the negligible one of the real
    sub-cell; rounding can also turn such a cell inside out (det J changes sign). Both kinds are
    listed in a warning; the cells are kept, and the mesh should be fixed at the source if the
    solve fails.

    Parameters
    ----------
    mesh : pyvista mesh with cell_data['materials'] (matrix = largest material ID)
    fiber_group : callable(material_ID) -> str, optional
        Name of the group a fiber cell belongs to. Defaults to a single group "fiber".
    degenerate_tol : float
        Cells with 0 <= cell_shape_ratio < degenerate_tol are reported as nearly collapsed;
        cells with a negative ratio are reported as inverted.

    Returns
    -------
    dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
        {group: {shape: (cells, cell_ids)}}, "matrix" first, then the fiber groups by name; shapes
        ordered by node count, e.g. {"matrix": {"tri": ..., "quad": ..., "polygon5": ...}, ...}.
        cells has shape (E, N) and cell_ids holds the mesh cell index of each row.
    """
    if fiber_group is None:
        fiber_group = lambda material_ID: "fiber"

    points = np.asarray(mesh.points)[:, :2]
    materials = np.asarray(mesh.cell_data['materials'])    # Fiber=0, matrix= largest in the 'material'
    matrix_ID = np.max(materials)

    groups = {}
    degenerate = []    # (cell id, shape ratio)
    inverted = []      # (cell id, shape ratio)
    for id, celltype in enumerate(np.asarray(mesh.celltypes)):
        vtk_nodes = mesh.get_cell(id).point_ids
        cell_nodes = reorder_cell_basix(mesh, vtk_nodes, celltype)
        ratio = cell_shape_ratio(points[vtk_nodes])
        if ratio < 0.0:
            inverted.append((id, ratio))
        elif ratio < degenerate_tol:
            degenerate.append((id, ratio))

        name = "matrix" if materials[id] == matrix_ID else fiber_group(materials[id])
        shape = cell_shape(celltype, len(vtk_nodes))
        cells, cell_ids = groups.setdefault(name, {}).setdefault(shape, ([], []))
        cells.append(cell_nodes)
        cell_ids.append(id)

    if degenerate:
        warnings.warn(
            f"{len(degenerate)} nearly collapsed cell(s) (shape ratio < {degenerate_tol:g}), kept in the model: "
            + ", ".join(f"{id} ({ratio:.1e})" for id, ratio in degenerate),
            stacklevel=2,
        )
    if inverted:
        warnings.warn(
            f"{len(inverted)} inverted cell(s) (det J changes sign inside the cell), kept in the model: "
            + ", ".join(f"{id} ({ratio:.1e})" for id, ratio in inverted),
            stacklevel=2,
        )

    return {
        name: {
            shape: (np.array(cells), np.array(cell_ids))
            for shape, (cells, cell_ids) in sorted(groups[name].items(), key=lambda s: (len(s[1][0][0]), s[0]))
        }
        for name in sorted(groups, key=lambda name: (name != "matrix", name))
    }


def tri_quad_arrays(shapes):
    """
    (tri_cells, quad_cells, tri_id, quad_id) of one group returned by sort_cells_by_material, for
    the tests that hard-code one tri and one quad batch per material. Raises if the group has
    other shapes, since those cells would otherwise be left out of the model.
    """
    other = sorted(set(shapes) - {"tri", "quad"})
    if other:
        raise ValueError(
            f"Mesh has {other} cells, which a tri/quad-only element setup would leave out; build the "
            "element batches by looping over sort_cells_by_material (see test_fea_solve_dmg_scan_JetSCI.py)."
        )
    empty = (np.array([]), np.array([]))
    tri_cells, tri_id = shapes.get("tri", empty)
    quad_cells, quad_id = shapes.get("quad", empty)
    return tri_cells, quad_cells, tri_id, quad_id


def split_matrix_fiber_cells(mesh, degenerate_tol=1e-8):
    """
    Matrix/fiber, tri/quad-only version of sort_cells_by_material.

    Returns
    -------
    (matrix_tri_cells, matrix_quad_cells, matrix_tri_id, matrix_quad_id,
     fiber_tri_cells,  fiber_quad_cells,  fiber_tri_id,  fiber_quad_id)
    """
    groups = sort_cells_by_material(mesh, degenerate_tol=degenerate_tol)
    return (*tri_quad_arrays(groups.get("matrix", {})), *tri_quad_arrays(groups.get("fiber", {})))


def find_print_cell_idx(print_cell_ID, batch_cell_ids):
    '''
    Parameters
    ----------
    print_cell_ID : Pyvista cell ID
    batch_cell_ids : list of the mesh cell IDs of each element batch, in batch order

    Returns
    -------
    (index of the cell within its batch, index of the batch)
    '''
    for fib_matrix_shape, cell_ids in enumerate(batch_cell_ids):
        hits = np.flatnonzero(cell_ids == print_cell_ID)
        if hits.size:
            return int(hits[0]), fib_matrix_shape
    raise ValueError(f"Cell {print_cell_ID} is in no element batch")
