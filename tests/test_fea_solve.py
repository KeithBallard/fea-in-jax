import meshio

import numpy as np

import fea_traditional as test
from helper import *

# from jax_smi import initialise_tracking
# initialise_tracking()

# Read in the mesh
mesh = meshio.read(get_mesh(f"polygon_mesh_{0.5}.vtk"))
points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
cells = np.array(mesh.cells[1].data, dtype=np.uint64)
print("# DoFs = ", 2 * points.shape[0])

# Sizes of arrays
U = 2  # number of solution components
V = points.shape[0]  # number of vertices
E = cells.shape[0]  # number of elements
M = 2  # number of material parameters
F = V * U  # number of DoFs
fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=2,
)
Q = get_quadrature(fe_type=fe_type)[0].shape[0] # number of quadrature points


# Define random Dirichlet boundary conditions
boundary_edges = np.array(mesh.cells[0].data, dtype=np.uint64)
boundary_points = np.unique(boundary_edges)
# An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
# Constrain every boundary point to have a random displacement
dirichlet_bcs = np.zeros((U * boundary_points.shape[0], 2), dtype=np.uint64)
for i in range(boundary_points.shape[0]):
    for j in range(U):
        dirichlet_bcs[U * i + j, 0] = i
        dirichlet_bcs[U * i + j, 1] = j
# Values of the Dirichlet boundary conditions matching 'dirichlet_bcs'
dirichlet_values = 0.001 * np.random.rand(dirichlet_bcs.shape[0])

# Set material properties at the quadrature point level randomly seeded such that
# E = [90e9, 100e9] and nu = 0.25
tmp_mat_params = np.random.rand(E, Q, M)
tmp_mat_params[..., 0] = 90e9 * tmp_mat_params[..., 0] + 10e9
tmp_mat_params[..., 1] = 0.25
mat_params_eqm = jnp.array(tmp_mat_params)

element_batches = [
    ElementBatch(
        fe_type=FiniteElementType(
            cell_type=CellType.triangle,
            family=ElementFamily.P,
            basis_degree=1,
            lagrange_variant=LagrangeVariant.equispaced,
            quadrature_type=QuadratureType.default,
            quadrature_degree=2,
        ),
        n_dofs_per_basis=2,
        connectivity_en=cells,
        constitutive_model=elastic_isotropic,
        material_params_eqm=mat_params_eqm,
        internal_state_eqi=jnp.zeros(shape=(E, Q, 0))
    )
]

# Solve the boundary value problem
u, residual, element_batches = solve_bvp(
    element_residual_func=linear_elasticity_residual,
    vertices_vd=points,
    element_batches=element_batches,
    u_0_g=jnp.zeros(shape=(V * U)),
    dirichlet_bcs=dirichlet_bcs,
    dirichlet_values=dirichlet_values,
    solver_options=SolverOptions(linear_solve_type=LinearSolverType.DIRECT_SPARSE_SOLVE_JNP),
)
print("|R| = ", jnp.linalg.norm(residual))
# print(residual)

# Make sure the solution matches at the Dirichlet BCs
dirichlet_dofs = U * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1]
assert jnp.isclose(u[dirichlet_dofs], dirichlet_values).all()

# Write output
mesh.point_data["u"] = u.reshape((points.shape[0], U))
mesh.write(get_output("test_fea_solve_out.vtk"))
