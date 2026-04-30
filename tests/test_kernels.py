import pytest

from kernels import *


def test_kernel_vectorized_implementations_match_loop_references():
    pygmsh = pytest.importorskip("pygmsh")

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0.0, 0.0],
                [1.0, -0.2],
                [1.1, 1.2],
                [0.1, 0.7],
            ],
            mesh_size=0.05,
        )
        mesh = geom.generate_mesh()

    points = np.array(mesh.points, dtype=np.float32)
    cells = np.array(mesh.cells[1].data, dtype=np.uint64)
    dims = Dimensions(N_gn=points.shape[0], N_ge=cells.shape[0])

    x_n = mesh_to_jax(points, cells)
    xi_qp = get_triangle_gauss_quadrature_4()[:, 0:2]
    W_qp = get_triangle_gauss_quadrature_4()[:, 2]
    phi_qp = triangle_basis_p1(xi_qp)
    dphi_dxi_qp = triangle_basis_p1_d_xi(xi_qp)
    u_n = jnp.array(np.random.rand(dims.N_ge, dims.N_n, dims.N_u))
    tmp_mat_params = np.random.rand(dims.N_ge, dims.N_qp, dims.N_mp)
    tmp_mat_params[..., 0] = 90e9 * tmp_mat_params[..., 0] + 10e9
    mat_params_qp = jnp.array(tmp_mat_params)

    x_n_alt = x_n.transpose((0, 2, 1))
    phi_qp_alt = phi_qp.T
    dphi_dxi_qp_alt = dphi_dxi_qp.transpose((1, 2, 0))

    x_qp_tensordot = k1_interp_node_to_quad_tensordot(x_n, phi_qp)
    x_qp_tensordot_alt = k1_interp_node_to_quad_tensordot_alt(x_n_alt, phi_qp_alt)
    x_qp_einsum_alt = k1_interp_node_to_quad_einsum_alt(x_n_alt, phi_qp_alt)
    x_qp_test = k1_interp_node_to_quad_sum(x_n, phi_qp, dims)
    assert jnp.array_equal(x_qp_tensordot, x_qp_test)
    assert jnp.array_equal(x_qp_tensordot_alt.transpose((0, 2, 1)), x_qp_test)
    assert jnp.array_equal(x_qp_einsum_alt.transpose((0, 2, 1)), x_qp_test)

    J_qp_tensordot = k3_param_to_global_jacobian_tensordot(x_n, dphi_dxi_qp)
    J_qp_tensordot_alt = k3_param_to_global_jacobian_tensordot_alt(
        x_n_alt, dphi_dxi_qp_alt
    )
    J_qp_test = k3_param_to_global_jacobian_sum(x_n, dphi_dxi_qp, dims)
    assert jnp.array_equal(J_qp_tensordot, J_qp_test)
    assert jnp.array_equal(J_qp_tensordot_alt.transpose((0, 3, 2, 1)), J_qp_test)

    G_qp_inv = k4_global_to_param_jacobian_inv(J_qp_tensordot)
    G_qp_inv_alt = k4_global_to_param_jacobian_inv_alt(J_qp_tensordot_alt)
    G_qp_test = k4_global_to_param_jacobian_loop(J_qp_tensordot, dims)
    assert jnp.isclose(G_qp_inv, G_qp_test).all()
    assert jnp.isclose(G_qp_inv_alt.transpose(0, 3, 2, 1), G_qp_test).all()

    det_J_qp = k5_calc_jacobian_det(J_qp_tensordot)
    det_J_qp_alt = k5_calc_jacobian_det_alt(J_qp_tensordot_alt)
    det_J_qp_test = k5_calc_jacobian_det_loop(J_qp_test, dims)
    assert jnp.isclose(det_J_qp, det_J_qp_test).all()
    assert jnp.isclose(det_J_qp_alt, det_J_qp_test).all()

    dphi_dx_qp_einsum = k6_basis_derivatives_global_einsum(G_qp_inv, dphi_dxi_qp)
    dphi_dx_qp_einsum_alt = k6_basis_derivatives_global_einsum_alt(
        G_qp_inv_alt, dphi_dxi_qp_alt
    )
    dphi_dx_qp_einsum_alt2 = k6_basis_derivatives_global_einsum_alt2(
        G_qp_inv.transpose((0, 1, 3, 2)), dphi_dxi_qp
    )
    dphi_dx_qp_test = k6_basis_derivatives_global_loop(G_qp_inv, dphi_dxi_qp, dims)
    assert jnp.isclose(dphi_dx_qp_einsum, dphi_dx_qp_test).all()
    assert jnp.isclose(dphi_dx_qp_einsum_alt, dphi_dx_qp_test).all()
    assert jnp.isclose(dphi_dx_qp_einsum_alt2, dphi_dx_qp_test).all()

    du_dx_qp_einsum = k7_grad_solution_global_einsum(dphi_dx_qp_einsum, u_n)
    du_dx_qp_test = k7_grad_solution_global_loop(dphi_dx_qp_einsum, u_n, dims)
    assert jnp.isclose(du_dx_qp_einsum, du_dx_qp_test).all()

    eps_qp = k8_strain(du_dx_qp_einsum)
    eps_qp_test = k8_strain_loop(du_dx_qp_einsum, dims)
    assert jnp.isclose(eps_qp, eps_qp_test).all()

    eps_voigt_qp = k8_strain_voigt(du_dx_qp_einsum)
    stress_qp = k9_stress_isotropic(mat_params_qp, eps_qp)
    stress_qp_test = k9_stress_isotropic_loop(mat_params_qp, eps_qp, dims)
    assert jnp.isclose(stress_qp, stress_qp_test, atol=100.0, rtol=1e-2).all()

    stress_voigt_qp = k9_stress_isotropic_voigt(mat_params_qp, eps_voigt_qp)
    assert stress_voigt_qp.shape[:2] == stress_qp.shape[:2]

    imbalance_qp = k10_grad_dphi_dx_stress(dphi_dx_qp_einsum, stress_qp)
    imbalance_qp_test = k10_grad_dphi_dx_stress_loop(
        dphi_dx_qp_einsum, stress_qp, dims
    )
    assert jnp.isclose(imbalance_qp, imbalance_qp_test).all()

    R_e = k11_residual(imbalance_qp, det_J_qp, W_qp)
    R_e_test = k11_residual_loop(imbalance_qp, det_J_qp, W_qp, dims)
    assert jnp.isclose(R_e, R_e_test, atol=100.0, rtol=1e-3).all()
