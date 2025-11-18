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
#  2 o-----------o 1 ^
#    |  \  2  /  |   |
#    | 3   o 4 1 |   2
#    |  /  0  \  |   |
#  3 o-----------o 0 V
#    ^ origin (0, 0)
mesh = meshio.Mesh(
    points=[
        [3.0, 0.0, 0.0],
        [3.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ],
    cells=[
        ("triangle", [[0, 4, 3], [1, 4, 0], [2, 4, 1], [3, 4, 2]]),
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


# Boundary Conditions
#  2 >o-----------o --> 1
#     |           |
#     |           |
#     |           |
#  3 >o-----------o --> 0
#     ^           ^
# An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
# Fixes left two points in x, fixes bottom two points in y, and moves right edge to the right
dirichlet_bcs = np.array(
    [[0, 0], [0, 1], [1, 0], [2, 0], [3, 0], [3, 1]], dtype=np.uint64
)
# Values of the Dirichlet boundary conditions matching 'dirichlet_bcs'
# Fixes bottom-left and moves top-right to the right by 1
dirichlet_values = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
print("dirichlet_bcs = \n", dirichlet_bcs)
print("dirichlet_values = ", dirichlet_values)

# Material properties
tmp_mat_params = np.zeros((E, Q, 2))
tmp_mat_params[..., 0] = 30e6
tmp_mat_params[..., 1] = 0.25
mat_params_eqp = jnp.array(tmp_mat_params)

# Setup element batches
element_batches = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=U,
        connectivity_en=cells[:1,:],
        constitutive_model=elastic_isotropic,
        material_params_eqm=mat_params_eqp[:1,:]
    ),
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=U,
        connectivity_en=cells[1:,:],
        constitutive_model=elastic_isotropic,
        material_params_eqm=mat_params_eqp[1:,:]
    ),
]

# Solve the boundary value problem
u, residual, new_internal_state_beqi = solve_bvp(
    element_residual_func=linear_elasticity_residual,
    vertices_vd=points,
    element_batches=element_batches,
    u_0_g=jnp.zeros(shape=(V * U)),
    dirichlet_bcs=dirichlet_bcs,
    dirichlet_values=dirichlet_values,
    solver_options=SolverOptions(
        linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
        linear_relative_tol=1e-8,
        linear_absolute_tol=0,
    ),
    plot_convergence=True
)

assert (u[0] - 1.0) < 1e-6 and (u[2] - 1.0) < 1e-6
assert abs(-(u[3] / 2.0) / (u[0] / 3.0) - 0.25) < 1e-6
assert abs(-(u[5] / 2.0) / (u[0] / 3.0) - 0.25) < 1e-6
assert u[1] == u[7] == 0.0
assert u[4] == u[6] == 0.0

# Write output
mesh.point_data["u"] = u.reshape((points.shape[0], U))
mesh.write(get_output("test_simple_fea_solve_out.vtk"))
