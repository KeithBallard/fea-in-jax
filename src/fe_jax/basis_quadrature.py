import numpy as np

from basix import (
    CellType,
    ElementFamily,
    LagrangeVariant,
    QuadratureType,
    PolysetType,
    create_element,
    make_quadrature,
)

from flax import struct
from typing import Any


@struct.dataclass
class FiniteElementType:
    """
    Defines the properties of a finite element formulation used to compute the basis
    functions and quadratures, see eval_basis_and_derivatives() and get_quadrature().
    """
    cell_type: CellType
    family: ElementFamily
    basis_degree: int
    lagrange_variant: LagrangeVariant
    quadrature_type: QuadratureType
    quadrature_degree: int


@struct.dataclass
class PolygonElementType:
    """
    Isoparametric n-gon element with Wachspress shape functions, following the Polygon element of
    PIGFEM (src/Polygon.cpp), which IGFEM uses for VTK_POLYGON integration cells.

    The reference element is the regular n-gon with vertex k at angle 2*pi*(k+1)/n on the unit
    circle (counter-clockwise), so element nodes must be listed counter-clockwise around the
    perimeter. Quadrature applies a triangle rule of `quadrature_degree` to each of the n
    triangles fanned from the origin (PIGFEM default: degree 3, i.e. 4 points per triangle).
    """
    n_vertices: int
    quadrature_degree: int = 3


# Triangle rules of PIGFEM (src/tri3.cpp) on the reference triangle (0,0), (1,0), (0,1); weights
# sum to 1/2. Degree 3 is the 4-point rule with a negative centroid weight, which differs from the
# Basix default.
_PIGFEM_TRI_RULES = {
    1: (np.array([[1/3, 1/3]]), np.array([0.5])),
    2: (np.array([[1/6, 1/6], [1/6, 2/3], [2/3, 1/6]]), np.full(3, 1/6)),
    3: (np.array([[1/3, 1/3], [0.2, 0.2], [0.2, 0.6], [0.6, 0.2]]),
        np.array([-27/96, 25/96, 25/96, 25/96])),
}


def polygon_reference_vertices(n_vertices: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vertices of the reference n-gon, shape (n, 2)."""
    angles = 2.0 * np.pi * np.arange(1, n_vertices + 1) / n_vertices
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _eval_wachspress(n_vertices: int, xi_qp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Wachspress shape functions on the regular reference n-gon and their derivatives, evaluated at
    interior points xi_qp (Q, 2). For a regular polygon the corner-triangle areas are all equal,
    so the weight of vertex k reduces to 1 / (A_{k-1} A_k), with A_k the signed area of the
    triangle (xi, p_k, p_{k+1}).
    """
    p = polygon_reference_vertices(n_vertices)
    p_next = np.roll(p, -1, axis=0)
    xi = np.asarray(xi_qp, dtype=np.float64)[:, None, :]    # (Q, 1, 2)

    # A_qk = 0.5 * cross(p_k - xi, p_{k+1} - xi) and its gradient with respect to xi
    a, b = p - xi, p_next - xi
    A_qk = 0.5 * (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])
    dA_kp = 0.5 * np.stack([p[:, 1] - p_next[:, 1], p_next[:, 0] - p[:, 0]], axis=1)

    A_prev_qk = np.roll(A_qk, 1, axis=1)
    dA_prev_kp = np.roll(dA_kp, 1, axis=0)
    w_qk = 1.0 / (A_prev_qk * A_qk)
    dw_qkp = -w_qk[..., None] * (dA_prev_kp / A_prev_qk[..., None] + dA_kp / A_qk[..., None])

    sum_w_q = np.sum(w_qk, axis=1)
    phi_qn = w_qk / sum_w_q[:, None]
    dphi_dxi_qnp = (dw_qkp - phi_qn[..., None] * np.sum(dw_qkp, axis=1)[:, None, :]) / sum_w_q[:, None, None]
    return phi_qn, dphi_dxi_qnp


def _polygon_quadrature(fe_type: PolygonElementType) -> tuple[np.ndarray, np.ndarray]:
    if fe_type.quadrature_degree in _PIGFEM_TRI_RULES:
        xi_t, W_t = _PIGFEM_TRI_RULES[fe_type.quadrature_degree]
    else:
        xi_t, W_t = make_quadrature(cell=CellType.triangle, degree=fe_type.quadrature_degree)
        xi_t, W_t = np.array(xi_t), np.array(W_t)

    # Map the triangle rule onto each sub-triangle (origin, p_k, p_{k+1})
    p = polygon_reference_vertices(fe_type.n_vertices)
    p_next = np.roll(p, -1, axis=0)
    xi = (xi_t[None, :, 0:1] * p[:, None, :] + xi_t[None, :, 1:2] * p_next[:, None, :]).reshape(-1, 2)
    det_k = p[:, 0] * p_next[:, 1] - p[:, 1] * p_next[:, 0]
    W = (det_k[:, None] * W_t[None, :]).ravel()
    return xi, W


def eval_basis_and_derivatives(
    fe_type: FiniteElementType | PolygonElementType, xi_qp: np.ndarray[Any, np.dtype[np.float64]]
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """
    Evaluates basis functions and its first derivatives at the given parametric points.
    Uses basis functions from the basix library, see https://github.com/FEniCS/basix for numbering
    documentation.

    Parameters
    ----------
    fe_type : defines the type of finite element for which to evaluate the basis functions
    xi_qp   : dense 2d-array with shape (Q, P)

    Returns
    -------
    phi      : 2d-array with shape (Q, N)
    dphi_dxi : 3d-array with shape (Q, N, P)
    """
    if isinstance(fe_type, PolygonElementType):
        return _eval_wachspress(fe_type.n_vertices, xi_qp)

    e = create_element(
        family=fe_type.family,
        celltype=fe_type.cell_type,
        degree=fe_type.basis_degree,
        lagrange_variant=fe_type.lagrange_variant,
    )
    r = np.array(e.tabulate(n=1, x=xi_qp))

    return (r[0, :, :, 0], r[1:3, :, :, 0].transpose((1, 2, 0)))


def get_quadrature(
    fe_type: FiniteElementType | PolygonElementType,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """
    Get quadrature scheme (points and weights) for a given finite element type. Uses quadrature
    from the basix library. For documentation, refer to:
    https://docs.fenicsproject.org/basix/main/python/demo/demo_quadrature.py.html.

    Parameters
    ----------
    fe_type : defines the type of finite element the quadrature is for

    Returns
    -------
    xi : 2d-array with shape (Q, P)
    W  : 1d-array with shape (Q,)
    """
    if isinstance(fe_type, PolygonElementType):
        return _polygon_quadrature(fe_type)

    quad_points, weights = make_quadrature(
        cell=fe_type.cell_type,
        degree=fe_type.quadrature_degree,
        rule=fe_type.quadrature_type,
    )
    return (np.array(quad_points), np.array(weights))
