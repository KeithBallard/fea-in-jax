import pygmsh

from kernels import *

# General notes:
# 1) It might be helpful to inherit from jax.array and add labels for axes to
#    help with debugging and enable a higher level description of operations.

# Mesh of a polygon
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

# Setup inputs
x_n = mesh_to_jax(points, cells)
xi_qp = get_triangle_gauss_quadrature_4()[:, 0:2]
W_qp = get_triangle_gauss_quadrature_4()[:, 2]
phi_qp = triangle_basis_p1(xi_qp)
dphi_dxi_qp = triangle_basis_p1_d_xi(xi_qp)
u_n = jnp.array(np.random.rand(dims.N_ge, dims.N_n, dims.N_u))
tmp_mat_params = np.random.rand(dims.N_ge, dims.N_qp, dims.N_mp)
tmp_mat_params[...,0] = 90e9 * tmp_mat_params[...,0] + 10e9
mat_params_qp = jnp.array(tmp_mat_params)
print("x_n", x_n.shape)  # , x_n)
print("xi_qp", xi_qp.shape)  # , xi_qp)
print("phi_qp", phi_qp.shape)  # , phi_qp)
print("dphi_dxi_qp", dphi_dxi_qp.shape)  # , dphi_dxi_qp)
print("u_n", u_n.shape)  # , u_n)
print("mat_params_qp", mat_params_qp.shape)  # , mat_params_qp)

# Alternative memory layouts (exploring how memory layout affects operations)
x_n_alt = x_n.transpose((0, 2, 1))
phi_qp_alt = phi_qp.T
dphi_dxi_qp_alt = dphi_dxi_qp.transpose((1, 2, 0))
print("x_n_alt", x_n_alt.shape)  # , x_n)
print("phi_qp_alt", phi_qp_alt.shape)  # , phi_qp)
print("dphi_dxi_qp_alt", dphi_dxi_qp_alt.shape)  # , dphi_dxi_qp)

###################################################################################################
# Kernel #1

print('\nKernel 1\n---------------\n')

x_qp_tensordot = k1_interp_node_to_quad_tensordot(x_n, phi_qp)
x_qp_tensordot_alt = k1_interp_node_to_quad_tensordot_alt(x_n_alt, phi_qp_alt)
x_qp_einsum_alt = k1_interp_node_to_quad_einsum_alt(x_n_alt, phi_qp_alt)
print("x_qp_tensordot", x_qp_tensordot.shape)  # , x_qp_tensordot)
print("x_qp_tensordot_alt", x_qp_tensordot_alt.shape)  # , x_qp_tensordot_alt)
print("x_qp_einsum_alt", x_qp_einsum_alt.shape)  # , x_qp_einsum_alt)

# Test results match classical loop version
x_qp_test = k1_interp_node_to_quad_sum(x_n, phi_qp, dims)
assert jnp.array_equal(x_qp_tensordot, x_qp_test)
assert jnp.array_equal(x_qp_tensordot_alt.transpose((0, 2, 1)), x_qp_test)
assert jnp.array_equal(x_qp_einsum_alt.transpose((0, 2, 1)), x_qp_test)

# Note: Rough benchmarking insights,
#       1) Changing the shape of x_n and phi_qp reduced the time required for the kernel by
#          40-50% (comparing X_tensordot to the alt versions).
#       2) For the alternate memory layout, tensordot and einsum had similar performance
#          per a call, but the einsum required many orders of magnitude longer time for JIT.

###################################################################################################
# Kernel #3

print('\nKernel 3\n---------------\n')

J_qp_tensordot = k3_param_to_global_jacobian_tensordot(x_n, dphi_dxi_qp)
J_qp_tensordot_alt = k3_param_to_global_jacobian_tensordot_alt(x_n_alt, dphi_dxi_qp_alt)
print("J_qp_tensordot", J_qp_tensordot.shape) # , J_qp_tensordot)
print("J_qp_tensordot_alt", J_qp_tensordot_alt.shape) # , J_qp_tensordot_alt)

# Test results match classical loop version
J_qp_test = k3_param_to_global_jacobian_sum(x_n, dphi_dxi_qp, dims)
assert jnp.array_equal(J_qp_tensordot, J_qp_test)
assert jnp.array_equal(J_qp_tensordot_alt.transpose((0, 3, 2, 1)), J_qp_test)

# Note: Rough benchmarking insight, the altnative memory layout was ~50% faster.

###################################################################################################
# Kernel #4

print('\nKernel 4\n---------------\n')

G_qp_inv = k4_global_to_param_jacobian_inv(J_qp_tensordot)
G_qp_inv_alt = k4_global_to_param_jacobian_inv_alt(J_qp_tensordot_alt)
print("G_qp_inv", G_qp_inv.shape) # , G_qp_inv)
print("G_qp_inv_alt", G_qp_inv_alt.shape) # , G_qp_inv_alt)

# Test results match classical loop version
G_qp_test = k4_global_to_param_jacobian_loop(J_qp_tensordot, dims)
print("G_qp_test", G_qp_test.shape) # , G_qp_test)
assert jnp.isclose(G_qp_inv, G_qp_test).all()
assert jnp.isclose(G_qp_inv_alt.transpose(0, 3, 2, 1), G_qp_test).all()

# Note: Rough benchmarking insight, both memory layouts were about the same.

###################################################################################################
# Kernel #5

print('\nKernel 5\n---------------\n')

det_J_qp = k5_calc_jacobian_det(J_qp_tensordot)
det_J_qp_alt = k5_calc_jacobian_det_alt(J_qp_tensordot_alt)
print("det_J_qp", det_J_qp.shape) # , det_J_qp)
print("det_J_qp_alt", det_J_qp_alt.shape) # , det_J_qp_alt)

# Test results match classical loop version
det_J_qp_test = k5_calc_jacobian_det_loop(J_qp_test, dims)
print("det_J_qp_test", det_J_qp_test.shape) # , det_J_qp_test)
assert jnp.isclose(det_J_qp, det_J_qp_test).all()
assert jnp.isclose(det_J_qp_alt, det_J_qp_test).all()

# Note: Rough benchmarking insight, the alternative memory layout does not have to be consistent
#       between J and G. If kernel 4 was modified to be a mix of the original and alt versions, 
#       then you can use k5_calc_jacobian_det with the alternative memory layouts and save time
#       since k5_calc_jacobian_det is ~3-4x faster than k5_calc_jacobian_det_alt.


###################################################################################################
# Kernel #6

print('\nKernel 6\n---------------\n')

dphi_dx_qp_einsum = k6_basis_derivatives_global_einsum(G_qp_inv, dphi_dxi_qp)
dphi_dx_qp_einsum_alt = k6_basis_derivatives_global_einsum_alt(G_qp_inv_alt, dphi_dxi_qp_alt)
dphi_dx_qp_einsum_alt2 = k6_basis_derivatives_global_einsum_alt2(G_qp_inv.transpose((0, 1, 3, 2)), dphi_dxi_qp)
print("dphi_dx_qp_einsum", dphi_dx_qp_einsum.shape) # , dphi_dx_qp_einsum)
print("dphi_dx_qp_einsum_alt", dphi_dx_qp_einsum_alt.shape) # , dphi_dx_qp_einsum_alt)
print("dphi_dx_qp_einsum_alt2", dphi_dx_qp_einsum_alt2.shape) # , dphi_dx_qp_einsum_alt2)

# Test results match classical loop version
dphi_dx_qp_test = k6_basis_derivatives_global_loop(G_qp_inv, dphi_dxi_qp, dims)
print("dphi_dx_qp_test", dphi_dx_qp_test.shape) # , dphi_dx_qp_test)
assert jnp.isclose(dphi_dx_qp_einsum, dphi_dx_qp_test).all()
assert jnp.isclose(dphi_dx_qp_einsum_alt, dphi_dx_qp_test).all()
assert jnp.isclose(dphi_dx_qp_einsum_alt2, dphi_dx_qp_test).all()

# Note: Rough benchmarking insight, the memory layouts I tried do not seem to affect this kernel.

###################################################################################################
# Kernel #7

print('\nKernel 7\n---------------\n')

du_dx_qp_einsum = k7_grad_solution_global_einsum(dphi_dx_qp_einsum, u_n)
print("du_dx_qp_einsum", du_dx_qp_einsum.shape) # , du_dx_qp_einsum)

du_dx_qp_test = k7_grad_solution_global_loop(dphi_dx_qp_einsum, u_n, dims)
print("du_dx_qp_test", du_dx_qp_test.shape) # , du_dx_qp_test)
assert jnp.isclose(du_dx_qp_einsum, du_dx_qp_test).all()

###################################################################################################
# Kernel #8

print('\nKernel 8\n---------------\n')

eps_qp = k8_strain(du_dx_qp_einsum)
print("eps_qp", eps_qp.shape) # , eps_qp)

eps_qp_test = k8_strain_loop(du_dx_qp_einsum, dims)
print("eps_qp_test", eps_qp_test.shape) # , eps_qp_test)
assert jnp.isclose(eps_qp, eps_qp_test).all()

eps_voigt_qp = k8_strain_voigt(du_dx_qp_einsum)
print("eps_voigt_qp", eps_voigt_qp.shape) # , eps_voigt_qp)

###################################################################################################
# Kernel #9

print('\nKernel 9\n---------------\n')

stress_qp = k9_stress_isotropic(mat_params_qp, eps_qp)
print("stress_qp", stress_qp.shape) # , stress_qp)

stress_qp_test = k9_stress_isotropic_loop(mat_params_qp, eps_qp, dims)
print("stress_qp_test", stress_qp_test.shape) #, stress_qp_test)
#diff = jnp.abs(stress_qp - stress_qp_test)
#lhs = (100.0 + 1e-3 * jnp.abs(stress_qp_test))
#print(f'compare {diff} <= {lhs}')
#print(stress_qp[diff > lhs])
#print(stress_qp_test[diff > lhs])
assert jnp.isclose(stress_qp, stress_qp_test, atol=100.0, rtol=1e-2).all()

stress_voigt_qp = k9_stress_isotropic_voigt(mat_params_qp, eps_voigt_qp)
print("stress_voigt_qp", stress_voigt_qp.shape) # , stress_voigt_qp)

###################################################################################################
# Kernel #10

print('\nKernel 10\n---------------\n')

imbalance_qp = k10_grad_dphi_dx_stress(dphi_dx_qp_einsum, stress_qp)
print("imbalance_qp", imbalance_qp.shape) # , imbalance_qp)


imbalance_qp_test = k10_grad_dphi_dx_stress_loop(dphi_dx_qp_einsum, stress_qp, dims)
print("imbalance_qp_test", imbalance_qp_test.shape) # , imbalance_qp_test)
assert jnp.isclose(imbalance_qp, imbalance_qp_test).all()

###################################################################################################
# Kernel #11

print('\nKernel 11\n---------------\n')

R_e = k11_residual(imbalance_qp, det_J_qp, W_qp)
print("R_e", R_e.shape) #, R_e)

R_e_test = k11_residual_loop(imbalance_qp, det_J_qp, W_qp, dims)
print("R_e_test", R_e_test.shape) #, R_e_test)
assert jnp.isclose(R_e, R_e_test, atol=100.0, rtol=1e-3).all()