from kernels import *
from helper import *

import jax.numpy as jnp
import jax
import jax.experimental.sparse as jsparse



def calculate_constitutive_matrix(
    mat_params_eqp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    """
    Parameters
    ----------
    mat_params_eqp : dense 3d-array with shape (N_ge, N_eqp, N_mp)

    Returns
    -------
    C   : dense 4d-array with shape (N_ge, N_eqp, N_eps, N_eps)
    """
    assert mat_params_eqp.shape[0] == dims.N_ge
    assert mat_params_eqp.shape[1] == dims.N_qp
    assert mat_params_eqp.shape[2] == dims.N_mp

    C = np.zeros((dims.N_ge, dims.N_qp, dims.N_eps, dims.N_eps))
    if dims.N_x == 1:  # 1D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_eqp[i, j, 0]
                C[i, j] = E
    elif dims.N_x == 2:  # 2D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_eqp[i, j, 0]
                nu = mat_params_eqp[i, j, 1]
                G = 0.5 * E / (1.0 + nu)
                zero = 0.0
                C[i, j] = jnp.linalg.inv(
                    np.array(
                        [
                            [1.0 / E, -nu / E, zero],
                            [-nu / E, 1.0 / E, zero],
                            [zero, zero, 1.0 / G],
                        ]
                    )
                )
    elif dims.N_x == 3:  # 3D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_eqp[i, j, 0]
                nu = mat_params_eqp[i, j, 1]
                G = 0.5 * E / (1.0 + nu)
                zero = 0.0
                C[i, j] = jnp.linalg.inv(
                    np.array(
                        [
                            [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                            [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                            [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                            [zero, zero, zero, 1.0 / G, zero, zero],
                            [zero, zero, zero, zero, 1.0 / G, zero],
                            [zero, zero, zero, zero, zero, 1.0 / G],
                        ]
                    )
                )
    return jnp.array(C)


def calculate_B_matrix(dphi_dx_eqp: jnp.ndarray, dims: Dimensions) -> jnp.ndarray:
    """
    Parameters
    ----------
    dphi_dx_eqp    : dense 4d-array with shape (N_ge, N_eqp, N_en, N_x)

    Returns
    -------
    B   : dense 4d-array with shape (N_ge, N_eqp, N_eps, N_en * N_x)
    """
    assert dphi_dx_eqp.shape[0] == dims.N_ge
    assert dphi_dx_eqp.shape[1] == dims.N_qp
    assert dphi_dx_eqp.shape[2] == dims.N_n
    assert dphi_dx_eqp.shape[3] == dims.N_x

    B = np.zeros((dims.N_ge, dims.N_qp, dims.N_eps, dims.N_n * dims.N_x))
    if dims.N_x == 1:  # 1D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                for k in range(dims.N_n):
                    B[i, j, 0, k + 0] = dphi_dx_eqp[i, j, k, 0]

    elif dims.N_x == 2:  # 2D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                for k in range(dims.N_n):
                    dphi_dx = dphi_dx_eqp[i, j, k, 0]
                    dphi_dy = dphi_dx_eqp[i, j, k, 1]

                    # Row 0 eps_xx
                    B[i, j, 0, dims.N_x * k + 0] = dphi_dx
                    B[i, j, 0, dims.N_x * k + 1] = 0.0
                    # Row 1 eps_yy
                    B[i, j, 1, dims.N_x * k + 0] = 0.0
                    B[i, j, 1, dims.N_x * k + 1] = dphi_dy
                    # Row 2 eps_xy
                    B[i, j, 2, dims.N_x * k + 0] = dphi_dy
                    B[i, j, 2, dims.N_x * k + 1] = dphi_dx

    elif dims.N_x == 3:  # 3D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                for k in range(dims.N_n):
                    dphi_dx = dphi_dx_eqp[i, j, k, 0]
                    dphi_dy = dphi_dx_eqp[i, j, k, 1]
                    dphi_dz = dphi_dx_eqp[i, j, k, 2]

                    # Row 0 eps_xx
                    B[i, j, 0, 3 * k + 0] = dphi_dx
                    B[i, j, 0, 3 * k + 1] = 0.0
                    B[i, j, 0, 3 * k + 2] = 0.0
                    # Row 1 eps_yy
                    B[i, j, 1, 3 * k + 0] = 0.0
                    B[i, j, 1, 3 * k + 1] = dphi_dy
                    B[i, j, 1, 3 * k + 2] = 0.0
                    # Row 2 eps_zz
                    B[i, j, 2, 3 * k + 0] = 0.0
                    B[i, j, 2, 3 * k + 1] = 0.0
                    B[i, j, 2, 3 * k + 2] = dphi_dz
                    # Row 3 eps_yz
                    B[i, j, 3, 3 * k + 0] = 0.0
                    B[i, j, 3, 3 * k + 1] = dphi_dz
                    B[i, j, 3, 3 * k + 2] = dphi_dy
                    # Row 4 eps_xz
                    B[i, j, 4, 3 * k + 0] = dphi_dz
                    B[i, j, 4, 3 * k + 1] = 0.0
                    B[i, j, 4, 3 * k + 2] = dphi_dx
                    # Row 5 eps_xy
                    B[i, j, 5, 3 * k + 0] = dphi_dy
                    B[i, j, 5, 3 * k + 1] = dphi_dx
                    B[i, j, 5, 3 * k + 2] = 0.0

    return jnp.array(B)


# @jax.jit
def calculate_residual(
    x_en: jnp.ndarray,
    dphi_dxi_eqp: jnp.ndarray,
    W_q: jnp.ndarray,
    mat_params_eqp: jnp.ndarray,
    cells_en: np.ndarray,
    assembly_map: jsparse.BCSR,
    u_gn: jnp.ndarray,
    dirichlet_elimination_gn: jnp.ndarray,
    dims: Dimensions,
) -> jnp.ndarray:
    """
    Compute the residual vector given the current solution and state information.

    Parameters
    ----------
    x_en           : dense 3d-array with shape (N_ge, N_en, N_x)
    phi_eqp        : dense 2d-array with shape (N_eqp, N_en)
    dphi_dxi_eqp   : dense 3d-array with shape (N_eqp, N_en, N_xi)
    W_q          : dense 2d-array with shape (N_ge, N_eqp)
    mat_params_eqp : dense 3d-array with shape (N_ge, N_eqp, N_mp)
    u_en           : dense 4d-array with shape (N_ge, N_en, N_u)

    Returns
    -------
    R_e  : dense 1d-array with shape (N_gn * N_u)
    """

    u_en = transform_global_unraveled_to_element_node(
        assembly_map, u_gn, N_ge=dims.N_ge, N_n=dims.N_n, N_u=dims.N_u
    )

    # x_eqp = k1_interp_node_to_quad_sum(x_en, phi_eqp, dims)
    J_eqp = k3_param_to_global_jacobian_sum(x_en, dphi_dxi_eqp, dims)
    G_eqp = k4_global_to_param_jacobian_loop(J_eqp, dims)
    det_J_eqp = k5_calc_jacobian_det_loop(J_eqp, dims)
    dphi_dx_eqp = k6_basis_derivatives_global_loop(G_eqp, dphi_dxi_eqp, dims)
    du_dx_eqp = k7_grad_solution_global_loop(dphi_dx_eqp, u_en, dims)
    eps_eqp = k8_strain_loop(du_dx_eqp, dims)
    stress_eqp = k9_stress_isotropic_loop(mat_params_eqp, eps_eqp, dims)
    imbalance_eqp = k10_grad_dphi_dx_stress_loop(dphi_dx_eqp, stress_eqp, dims)
    R_e = k11_residual_loop(imbalance_eqp, det_J_eqp, W_q, dims)

    B_eqp = calculate_B_matrix(dphi_dx_eqp, dims)
    C_eqp = calculate_constitutive_matrix(mat_params_eqp, dims)
    eps_voigt = rank2_tensor_to_voigt(eps_eqp)

    # Check that the strain is being computed correctly
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            eps_via_B = np.dot(B_eqp[i, j], u_en[i].reshape(np.prod(u_en[i].shape)))
            assert np.isclose(eps_via_B, eps_voigt[i, j]).all()

    R_gn = np.zeros((dims.N_gn * dims.N_u,), dtype=np.float32)
    for i in range(dims.N_ge):
        dof_map = [int(dims.N_u * n + m) for n in cells_en[i] for m in range(dims.N_u)]
        R_element = np.zeros((len(dof_map),))
        for j in range(dims.N_qp):
            R_element += (
                np.dot(B_eqp[i, j].T, np.dot(C_eqp[i, j], eps_voigt[i, j]))
                * det_J_eqp[i, j]
                * W_q[j]
            )
        R_gn[np.ix_(dof_map)] += R_element

    # Zero out terms corresponding to Dirichlet BCs
    R_gn = jnp.multiply(dirichlet_elimination_gn, jnp.array(R_gn))

    return R_gn


def calculate_jacobian(
    x_en: jnp.ndarray,
    dphi_dxi_eqp: jnp.ndarray,
    W_q: jnp.ndarray,
    mat_params_eqp: jnp.ndarray,
    cells_en: np.ndarray,
    dims: Dimensions,
):
    J_eqp = k3_param_to_global_jacobian_sum(x_en, dphi_dxi_eqp, dims)
    G_eqp = k4_global_to_param_jacobian_loop(J_eqp, dims)
    det_J_eqp = k5_calc_jacobian_det_loop(J_eqp, dims)
    dphi_dx_eqp = k6_basis_derivatives_global_loop(G_eqp, dphi_dxi_eqp, dims)

    # Compute element stiffness matrices
    B_eqp = calculate_B_matrix(dphi_dx_eqp, dims)
    C_eqp = calculate_constitutive_matrix(mat_params_eqp, dims)

    # Compute the element level stiffness matrices and assemble them into
    # the global matrix
    K_global = np.zeros((dims.N_gn * dims.N_u, dims.N_gn * dims.N_u), dtype=np.float32)
    K_elements = []
    for i in range(dims.N_ge):
        dof_map = [int(dims.N_u * n + m) for n in cells_en[i] for m in range(dims.N_u)]
        K_element = np.zeros((len(dof_map), len(dof_map)))
        for j in range(dims.N_qp):
            K_element += (
                np.dot(np.dot(B_eqp[i, j].T, C_eqp[i, j]), B_eqp[i, j])
                * det_J_eqp[i, j]
                * W_q[j]
            )
        K_global[np.ix_(dof_map, dof_map)] += K_element
        K_elements.append(K_element)

    return (K_global, K_elements)


if __name__ == "__main__":

    #
    # This test is based on the problem shown in the PDF in the repo entitled:
    # "2D Triangular Elements.pdf"
    #

    import meshio

    # Refer to "4.9 Example" in "2D Triangular Elements.pdf"
    # The mesh:
    #    <-----3----->
    #  3 o-----------o 2 ^
    #    |   2    /  |   |
    #    |     /     |   2
    #    |  /     1  |   |
    #  4 o-----------o 1 V
    #    ^ origin (0, 0)
    mesh = meshio.Mesh(
        points=[
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        cells=[
            ("triangle", [[0, 1, 3], [2, 3, 1]]),
        ],
    )

    points = np.array(mesh.points, dtype=np.float64)
    cells = np.array(mesh.cells[0].data, dtype=np.uint64)

    assembly_map = mesh_to_sparse_assembly_map(points.shape[0], cells)
    n_cell_per_vert = get_n_cells_per_vert(points, cells)

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=3,
    )

    x_en = mesh_to_jax(points, cells)
    xi_qp, W_q = get_quadrature(fe_type=fe_type)
    phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)

    dims = Dimensions(N_qp=W_q.shape[0], N_gn=points.shape[0], N_ge=cells.shape[0])

    # Refer to "4.9 Example" in "2D Triangular Elements.pdf"
    np_mat_params = np.zeros((dims.N_ge, dims.N_qp, dims.N_mp))
    np_mat_params[..., 0] = 30e6
    np_mat_params[..., 1] = 0.25
    mat_params_eqp = jnp.array(np_mat_params)

    C_eqp = calculate_constitutive_matrix(mat_params_eqp, dims)
    # Refer to Eqn. 4.14 in "2D Triangular Elements.pdf"
    assert jnp.isclose(
        C_eqp[0][0],
        jnp.array([[3.2e7, 0.8e7, 0.0], [0.8e7, 3.2e7, 0.0], [0.0, 0.0, 1.2e7]]),
    ).all()

    J_eqp = k3_param_to_global_jacobian_sum(x_en, jnp.array(dphi_dxi_qnp), dims)
    print("J_eqp", J_eqp.shape, J_eqp)
    # Analytical derivatives for element 1, refer to Eqns. 4.36 - 4.39
    dx_dxi = x_en[0][0][0] - x_en[0][2][0]
    dx_deta = x_en[0][1][0] - x_en[0][2][0]
    dy_dxi = x_en[0][0][1] - x_en[0][2][1]
    dy_deta = x_en[0][1][1] - x_en[0][2][1]
    J_e_1_test = jnp.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
    print("J_e_1_test", J_e_1_test.shape, "\n", J_e_1_test)
    assert jnp.isclose(J_eqp[0][0], J_e_1_test).all()

    det_J_eqp = k5_calc_jacobian_det_loop(J_eqp, dims)
    # Refer to section "For Element 1/2" in "2D Triangular Elements.pdf"
    assert jnp.isclose(det_J_eqp[:][0], jnp.array([6.0, 6.0])).all()

    G_eqp = k4_global_to_param_jacobian_loop(J_eqp, dims)
    # print("G_eqp", G_eqp.shape, "\n", G_eqp[0][0])

    dphi_dx_eqp = k6_basis_derivatives_global_loop(G_eqp, jnp.array(dphi_dxi_qnp), dims)
    B_eqp = calculate_B_matrix(dphi_dx_eqp, dims)
    # print("B_eqp", B_eqp.shape, "\n", B_eqp[0][0])
    # Refer to section "For Element 1" in "2D Triangular Elements.pdf"
    assert jnp.isclose(
        B_eqp[0][0],
        jnp.array(
            [
                [2.0 / 6.0, 0.0, 0.0, 0.0, -2.0 / 6.0, 0.0],
                [0.0, -3.0 / 6.0, 0.0, 3.0 / 6.0, 0.0, 0.0],
                [-3.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 0.0, 0.0, -2.0 / 6.0],
            ]
        ),
    ).all()
    # Refer to section "For Element 2" in "2D Triangular Elements.pdf"
    assert jnp.isclose(
        B_eqp[1][0],
        jnp.array(
            [
                [-2.0 / 6.0, 0.0, 0.0, 0.0, 2.0 / 6.0, 0.0],
                [0.0, 3.0 / 6.0, 0.0, -3.0 / 6.0, 0.0, 0.0],
                [3.0 / 6.0, -2.0 / 6.0, -3.0 / 6.0, 0.0, 0.0, 2.0 / 6.0],
            ]
        ),
    ).all()

    K_global, K_elements = calculate_jacobian(
        x_en=x_en,
        dphi_dxi_eqp=jnp.array(dphi_dxi_qnp),
        W_q=jnp.array(W_q),
        mat_params_eqp=mat_params_eqp,
        cells_en=cells,
        dims=dims,
    )
    # Note: the matrix is expected to be off by a factor of 2 since the pdf includes
    #       a scalar t = 0.5 (for thickness), refer to Eqn. 4.66.
    # Refer to section "For Element 1" in "2D Triangular Elements.pdf"
    jnp.isclose(
        K_elements[0] / 2.0,
        jnp.array(
            [
                [0.983, -0.5, -0.45, 0.2, -0.533, 0.3],
                [-0.5, 1.4, 0.3, -1.2, 0.2, -0.2],
                [-0.45, 0.3, 0.45, 0, 0, -0.3],
                [0.2, -1.2, 0, 1.2, -0.2, 0],
                [-0.533, 0.2, 0, -0.2, 0.533, 0],
                [0.3, -0.2, -0.3, 0, 0, 0.2],
            ]
        ),
    ).all()
    # Refer to section "For Element 2" in "2D Triangular Elements.pdf"
    jnp.isclose(
        K_elements[1] / 2.0,
        jnp.array(
            [
                [0.983, -0.5, -0.45, 0.2, -0.533, 0.3],
                [-0.5, 1.4, 0.3, -1.2, 0.2, -0.2],
                [-0.45, 0.3, 0.45, 0, 0, -0.3],
                [0.2, -1.2, 0, 1.2, -0.2, 0],
                [-0.533, 0.2, 0, -0.2, 0.533, 0],
                [0.3, -0.2, -0.3, 0, 0, 0.2],
            ]
        ),
    ).all()

    # Make sure a Jacobian matrix compute via finite difference is close to the exact
    # to ensure the residual calculation is consistent.
    u_gn_1 = jnp.array(np.random.random(dims.N_gn * dims.N_u), dtype=jnp.float32)
    R_gn_1 = calculate_residual(
        x_en=x_en,
        dphi_dxi_eqp=jnp.array(dphi_dxi_qnp),
        W_q=jnp.array(W_q),
        mat_params_eqp=mat_params_eqp,
        cells_en=cells,
        assembly_map=assembly_map,
        u_gn=u_gn_1,
        dirichlet_elimination_gn=jnp.ones_like(u_gn_1),
        dims=dims,
    )

    K_u_prod = jnp.dot(K_global, u_gn_1)
    print("K_u_prod", K_u_prod.shape, "\n", K_u_prod)
    print("R_gn_1", R_gn_1.shape, "\n", R_gn_1)
    assert jnp.isclose(K_u_prod, R_gn_1).all()
