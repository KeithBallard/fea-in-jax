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

# NOTE: commenting out the U * V term for now
mpcs = [
    MultiPointConstraint(
        dep_dof=U * right_points[i],
        indep_dofs=[U * left_points[i]],  # , U * V],
        factors=[1.0],  # , max_xy[0] - min_xy[0]],
    )
    for i in range(len(left_points))
]

right_u_val = (max_xy[0] - min_xy[0]) / 100.0
dirichlet_constraints = [
    DirichletConstraint(dep_dof=U * i, value=right_u_val) for i in left_points
] + [DirichletConstraint(dep_dof=U * i + 1, value=0.0) for i in bottom_points]

print(mpcs[0])


# Extract cells for each subdomain
matrix_cells = cells[cell_domain_ids == 0]
fiber_cells = cells[cell_domain_ids == 1]

# Set material properties
matrix_mat_params = jnp.array([3.45e9, 0.35])  # E  # nu
fiber_mat_params = jnp.array(
    [26e9, 26e9, 0.7218543046357615, 7.55e9]  # E_xx  # E_yy  # nu_xy  # G_xy
)

element_batches = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=2,
        connectivity_en=matrix_cells,
        constitutive_model=elastic_isotropic,
        material_params=matrix_mat_params,
    ),
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=2,
        connectivity_en=fiber_cells,
        constitutive_model=elastic_orthotropic,
        material_params=fiber_mat_params,
    ),
]

u_0 = jnp.zeros(shape=(V * U))

# Solve the boundary value problem
u, residual, element_batches = solve_bvp(
    element_residual_func=linear_elasticity_residual,
    vertices_vd=points,
    element_batches=element_batches,
    u_0_g=u_0,
    dirichlet_bcs=dirichlet_constraints,
    multipoint_constraints=mpcs,
    solver_options=SolverOptions(
        linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        linear_precond_type=PreconditionerType.JACOBI,
        # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
    ),
)
print("|R| = ", jnp.linalg.norm(residual))
# print(residual)

# Make sure the solution matches at the Dirichlet BCs
dirichlet_dofs = U * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1]
assert jnp.isclose(u[dirichlet_dofs], dirichlet_values).all()

# Write output
mesh.point_data["u"] = u.reshape((points.shape[0], U))
mesh.write(get_output(Path(__file__).stem + ".vtk"))
