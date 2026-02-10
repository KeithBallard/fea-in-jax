from helper import *

jax.config.update("jax_enable_x64", True)

import jax.extend

print(jax.extend.backend.get_backend().platform)

# from jax_smi import initialise_tracking
# initialise_tracking()

points = np.linspace(0, 1, 2, dtype=np.float32).reshape((-1, 1))
cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.uint64)
cell_domain_ids = np.zeros(cells.shape[0], dtype=np.int64)

# Sizes of arrays
U = 1  # number of solution components
V = points.shape[0]  # number of vertices
E = cells.shape[0]  # number of elements
F = V * U  # number of DoFs
fe_type = FiniteElementType(
    cell_type=CellType.interval,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=2,
)
Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points

print("# DoFs = ", F)

# Periodic boundary conditions
mpcs = [
    #    MultiPointConstraint(
    #        dep_dof=F - 1,
    #        indep_dofs=[0],
    #        factors=[1.0],
    #    )
]

dirichlet_constraints = [
    DirichletConstraint(dep_dof=0, value=0.0),
    DirichletConstraint(dep_dof=1, value=0.1),
]

print(mpcs)


# Set material properties
matrix_mat_params = jnp.array([3.45e9])  # E
element_batches = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=1,
        connectivity_en=cells,
        constitutive_model=elastic_isotropic,
        material_params=matrix_mat_params,
    )
]

# Solve the boundary value problem
u, residual, element_batches = solve_bvp(
    element_residual_func=linear_elasticity_residual,
    vertices_vd=points,
    element_batches=element_batches,
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

print(f"u = {u}")

# Write output
mesh = meshio.Mesh(
    points,
    [("line", cells)],
    point_data={"u": u.reshape((points.shape[0], U))},
)
mesh.write(get_output(Path(__file__).stem + ".vtk"))
