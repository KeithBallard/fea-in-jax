import meshio

import fea_traditional as test
from helper import *

import numpy as np

# General notes:
# 1) It might be helpful to inherit from jax.array and add labels for axes to
#    help with debugging and enable a higher level description of operations.


# mesh = meshio.read(f"test_meshes/polygon_mesh_{0.05}.vtk")
# points = np.array(mesh.points, dtype=np.float32)
# cells = np.array(mesh.cells[1].data, dtype=np.uint64)

# Make the mesh:
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
mesh.write(get_output("two_tri.vtk"))
points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
cells = np.array(mesh.cells[0].data, dtype=np.uint64)

# Sizes of arrays
U = 2  # number of solution components
V = points.shape[0]  # number of vertices
E = cells.shape[0]  # number of elements
F = V * U  # number of DoFs
fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=3,
)
Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points

# Material properties
tmp_mat_params = np.zeros((E, Q, 2))
tmp_mat_params[..., 0] = 30e6
tmp_mat_params[..., 1] = 0.25
mat_params_eqp = jnp.array(tmp_mat_params)

# Setup element batches
element_batches = [
    ElementBatch(
        fe_type=fe_type,
        connectivity_en=cells,
        constitutive_model=elastic_isotropic,
        material_params_eqm=mat_params_eqp,
        internal_state_eqi=jnp.zeros(shape=(cells.shape[0], Q, 0)),
    ),
]


xi_bqp, W_bq = zip(*[get_quadrature(fe_type=b.fe_type) for b in element_batches])
W_bq = list(W_bq)
phi_bqn, dphi_dxi_bqnp = zip(
    *[
        eval_basis_and_derivatives(fe_type=b.fe_type, xi_qp=xi_bqp[i])
        for i, b in enumerate(element_batches)
    ]
)
dphi_dxi_bqnp = list(dphi_dxi_bqnp)

material_params_beqm = [b.material_params_eqm for b in element_batches]
internal_state_beqi = [b.internal_state_eqi for b in element_batches]
x_bend = [
    mesh_to_jax(vertices=points, cells=b.connectivity_en)
    for b in element_batches
]

# Structures for mapping between cell-level arrays and global arrays
assembly_map_b = [
    mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
    for b in element_batches
]

# Initial solution
# Global unraveled vector
u_0_g = jnp.array(0.001 * np.random.rand(V * U))
print("u_0_g", u_0_g.shape, u_0_g)
# Element-node representation
u_0_n = transform_global_unraveled_to_element_node(
    assembly_map_b, u_0_g, E=E, V=V, U=U
)
print("u_0_n", u_0_n.shape, u_0_n)
# Converted back to global unraveled to ensure transforms work as expected
u_0_g_test = transform_element_node_to_global_unraveled_nosum(
    assembly_map_b, u_0_n
)
print("u_0_g_test", u_0_g_test.shape, u_0_g_test)
assert jnp.isclose(u_0_g, u_0_g_test).all()

exit()
"""
# Function that produces R(u)
residual_func = lambda u: calculate_residual(
    u_gn=u,
    x_en=x_n,
    dphi_dxi_eqp=dphi_dxi_qp,
    W_eqp=W_qp,
    mat_params_eqp=mat_params_qp,
    assembly_map=assembly_map,
    dirichlet_values_gn=jnp.zeros_like(u),
    dirichlet_mask_gn=jnp.zeros_like(u),
    N_ge=dims.N_ge,
    N_n=dims.N_n,
    N_u=dims.N_u,
)
R = residual_func(u_0_g)
print("R", R.shape, R)

R_test = test.calculate_residual(
    x_en=x_n,
    dphi_dxi_eqp=dphi_dxi_qp,
    W_eqp=W_qp,
    mat_params_eqp=mat_params_qp,
    cells_en=cells,
    assembly_map=assembly_map,
    u_gn=u_0_g,
    dirichlet_elimination_gn=jnp.ones_like(u_0_g),
    dims=dims,
)
print("R_test", R_test.shape, R_test)
assert jnp.isclose(R, R_test).all()

# Function that produces J(u) * z
jacobian_vector_product = lambda z: jax.jvp(
    residual_func,
    (u_0_g,),
    (z,),
)[1]

# Compute J(u_0) * u_0
jax_jacobian_u_prod = jacobian_vector_product(u_0_g)

jax_jacobian = jax.jacfwd(residual_func)(u_0_g)
print("jax_jacobian", jax_jacobian.shape)

# Produces J(u)
test_jacobian, test_element_J = test.calculate_jacobian(
    x_en=x_n,
    dphi_dxi_eqp=dphi_dxi_qp,
    W_eqp=W_qp,
    mat_params_eqp=mat_params_qp,
    cells_en=cells,
    dims=dims,
)
print("test_jacobian", test_jacobian.shape, test_jacobian)
# print('test_element_J', test_element_J)

# Compute J(u_0) * u_0
test_jacobian_u_prod = jnp.array(np.dot(test_jacobian, u_0_g))

print("jax_jacobian_u_prod", jax_jacobian_u_prod)
print("test_jacobian_u_prod", test_jacobian_u_prod)

assert jnp.isclose(jax_jacobian_u_prod, test_jacobian_u_prod).all()
"""