"""
Utilities for meshes produced by the IGFEM code (PIGFEM, run through run_code.sh).

An IGFEM output mesh is the integration (sub-cell) mesh of the cut background triangulation:
TRI3/QUAD4 cells, a 'materials' cell field in which the matrix has the largest material ID and
each fiber has its own ID, and enrichment nodes stored as ordinary points. These functions
sort its cells into matrix/fiber element batches with Basix node ordering, and remove cells
that the IGFEM VTK writer collapsed to zero area by rounding coordinates to 6 significant digits.
"""

import numpy as np


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


# 2x2 Gauss points on the unit square, matching the quad quadrature_degree=2 used in the tests
_GAUSS_2X2_1D = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))


def cell_shape_ratio(coords):
    """
    Measures how close a TRI3 or QUAD4 cell is to collapsing, independently of its size.

    Element stiffness scales like (edge length)^2 / det(J), so a cell whose det(J) is tiny
    compared with its squared edge length makes the global system singular or indefinite,
    whereas a cell that is merely small does not. The ratio is evaluated at the quadrature
    points the solver actually uses (centroid for TRI3, 2x2 Gauss for QUAD4).

    Parameters
    ----------
    coords : ndarray[float, (N, 2)]
        Node coordinates in VTK order (perimeter order for quads).

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
        raise ValueError(f"Unsupported element with {len(coords)} nodes")

    orientation = 1.0 if np.sum(dets) >= 0.0 else -1.0
    return float(np.min(orientation * dets) / longest_edge_sq)


def sort_cells_by_material(mesh, fiber_group=None, degenerate_tol=1e-8, verbose=True):
    """
    Splits the TRI3/QUAD4 cells of an IGFEM mesh into material groups, reorders their nodes to
    the Basix convention, and drops collapsed cells.

    IGFEM writes node coordinates with 6 significant digits, so integration sub-cells that are
    only ~1e-7 across can be rounded into collinear (zero-area) cells. Such a cell has an
    effectively infinite, randomly signed stiffness instead of the negligible one of the real
    sub-cell, so it is removed. The mesh points are left untouched (node numbering, boundary
    node lists and VTK output all index into them); an error is raised if removing a cell would
    leave one of its nodes attached to no other cell.

    Parameters
    ----------
    mesh : pyvista mesh with cell_data['materials'] (matrix = largest material ID)
    fiber_group : callable(material_ID) -> str, optional
        Name of the group a fiber cell belongs to. Defaults to a single group "fiber".
    degenerate_tol : float
        Cells with cell_shape_ratio below this are dropped.
    verbose : bool
        Print the cells that were dropped.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        For "matrix" and each fiber group: (tri_cells, quad_cells, tri_id, quad_id).
    """
    if fiber_group is None:
        fiber_group = lambda material_ID: "fiber"

    points = np.asarray(mesh.points)[:, :2]
    materials = np.asarray(mesh.cell_data['materials'])    # Fiber=0, matrix= largest in the 'material'
    matrix_ID = np.max(materials)

    groups = {}
    dropped = []    # (cell id, shape ratio, node ids)
    for id, celltype in enumerate(mesh.celltypes):
        if celltype not in (5, 9):  # Triangle, Quad
            continue
        vtk_nodes = mesh.get_cell(id).point_ids
        ratio = cell_shape_ratio(points[vtk_nodes])
        if ratio < 0.0:
            raise ValueError(f"Cell {id} is inverted (det J changes sign, shape ratio {ratio:.2e}).")
        if ratio < degenerate_tol:
            dropped.append((id, ratio, vtk_nodes))
            continue

        name = "matrix" if materials[id] == matrix_ID else fiber_group(materials[id])
        tri_cells, quad_cells, tri_id, quad_id = groups.setdefault(name, ([], [], [], []))
        cell_nodes = reorder_cell_basix(mesh, vtk_nodes)
        if celltype == 5:
            tri_cells.append(cell_nodes)
            tri_id.append(id)
        else:
            quad_cells.append(cell_nodes)
            quad_id.append(id)

    if dropped:
        used = np.zeros(points.shape[0], dtype=bool)
        for tri_cells, quad_cells, _, _ in groups.values():
            for cell_nodes in tri_cells + quad_cells:
                used[cell_nodes] = True
        orphans = sorted({n for _, _, nodes in dropped for n in nodes if not used[n]})
        if orphans:
            raise ValueError(
                f"Dropping degenerate cells {[d[0] for d in dropped]} would leave nodes {orphans} "
                "attached to no element; the mesh needs to be fixed at the source."
            )
        if verbose:
            print(f"Dropped {len(dropped)} degenerate cell(s) (shape ratio < {degenerate_tol:g}): "
                  + ", ".join(f"{id} ({ratio:.1e})" for id, ratio, _ in dropped))

    return {
        name: tuple(np.array(x) for x in group)
        for name, group in groups.items()
    }


def split_matrix_fiber_cells(mesh, degenerate_tol=1e-8, verbose=True):
    """
    Matrix/fiber version of sort_cells_by_material.

    Returns
    -------
    (matrix_tri_cells, matrix_quad_cells, matrix_tri_id, matrix_quad_id,
     fiber_tri_cells,  fiber_quad_cells,  fiber_tri_id,  fiber_quad_id)
    """
    groups = sort_cells_by_material(mesh, degenerate_tol=degenerate_tol, verbose=verbose)
    empty = tuple(np.array([]) for _ in range(4))
    return (*groups.get("matrix", empty), *groups.get("fiber", empty))


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
