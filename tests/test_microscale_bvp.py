from helper import *

jax.config.update("jax_enable_x64", True)

import jax.extend
print(jax.extend.backend.get_backend().platform)

# from jax_smi import initialise_tracking
# initialise_tracking()

# Read in the mesh
mesh = meshio.read(get_mesh(f"microscale_2D_r0.vtk"))
points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
cells = np.array(mesh.cells[0].data, dtype=np.uint64)
mesh.cell_data["DomainIDs"][0] = np.array(
    mesh.cell_data["DomainIDs"][0], dtype=np.int64
)
cell_domain_ids = mesh.cell_data["DomainIDs"][0].flatten()
print("# DoFs = ", 2 * points.shape[0])

# Sizes of arrays
U = 2  # number of solution components
V = points.shape[0]  # number of vertices
E = cells.shape[0]  # number of elements
F = V * U  # number of DoFs
fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=2,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=3,
)
Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points

# Define node sets
min_xy = np.min(points, axis=0)
max_xy = np.max(points, axis=0)
left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]

# Boundary conditions:
# - Fix left nodes along x-direction
# - Fix right nodes such that the model is subjected to 1% strain along x-axis
# - Fix bottom nodes along y-direction
# - Fix top nodes along y-direction
dirichlet_bcs, dirichlet_values = build_dirichlet_arrays_from_lists(
    point_indices=[left_points, right_points, bottom_points, top_points],
    components=[0, 0, 1, 1],
    values=[0.0, (max_xy[0] - min_xy[0]) / 100.0, 0.0, 0.0],
)

# Extract cells for each subdomain
matrix_cells = cells[cell_domain_ids == 0]
fiber_cells = cells[cell_domain_ids == 1]


# Set material properties
@jax.jit
def get_properties():
    # Neat 5220 Epoxy
    matrix_mat_params_eqm = jnp.zeros(shape=(matrix_cells.shape[0], Q, 2))
    matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(3.45e9)  # E
    matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 1].set(0.35)  # nu
    # IM7 Fiber
    fiber_mat_params_eqm = jnp.zeros(shape=(fiber_cells.shape[0], Q, 4))
    fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(26e9)  # E_xx
    fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(26e9)  # E_yy
    fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(
        0.7218543046357615
    )  # nu_xy
    fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(7.55e9)  # G_xy
    return (matrix_mat_params_eqm, fiber_mat_params_eqm)

matrix_mat_params_eqm, fiber_mat_params_eqm = get_properties()
print(fiber_mat_params_eqm[0, 0, :])

# 3D properties, in case a 3D test is needed
# fiber_mat_params_eqm[:, :, 0] = 26e9 # E_xx
# fiber_mat_params_eqm[:, :, 1] = 26e9 # E_yy
# fiber_mat_params_eqm[:, :, 2] = 276e9 # E_zz
# fiber_mat_params_eqm[:, :, 3] = 0.7218543046357615 # nu_xy
# nu_zx = 0.292 # nu_zy
# fiber_mat_params_eqm[:, :, 4] = nu_zx * 26e9 / 276e9 # nu_yz
# fiber_mat_params_eqm[:, :, 5] = nu_zx * 26e9 / 276e9 # nu_xz
# fiber_mat_params_eqm[:, :, 6] = 7.55e9 # G_xy
# fiber_mat_params_eqm[:, :, 7] = 20.7e9 # G_yz
# fiber_mat_params_eqm[:, :, 8] = 20.7e9 # G_xz

element_batches = [
    ElementBatch(
        fe_type=fe_type,
        connectivity_en=matrix_cells,
        constitutive_model=elastic_isotropic,
        material_params_eqm=matrix_mat_params_eqm,
        internal_state_eqi=jnp.zeros(shape=(matrix_cells.shape[0], Q, 0)),
    ),
    ElementBatch(
        fe_type=fe_type,
        connectivity_en=fiber_cells,
        constitutive_model=elastic_orthotropic,
        material_params_eqm=fiber_mat_params_eqm,
        internal_state_eqi=jnp.zeros(shape=(fiber_cells.shape[0], Q, 0)),
    ),
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
mesh.write(get_output("test_microscale_bvp_out.vtk"))
