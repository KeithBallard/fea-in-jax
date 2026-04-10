from helper import *

jax.config.update("jax_enable_x64", True)

import jax.extend

print(jax.extend.backend.get_backend().platform)

# from jax_smi import initialise_tracking
# initialise_tracking()

n_elements = 4
points = np.linspace(0, 1, n_elements + 1, dtype=np.float32).reshape((-1, 1))
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

# Set material properties
matrix_mat_params = jnp.array([1.0e9])  # E

# Set boundary conditions (first endpoint stays fixed, final endpoint goes from 1.0 -> 1.2
bcs = ([DirichletBC(bc_type = BCType.NODE,component=0, index=0,value=0.0),
        DirichletBC(bc_type=BCType.NODE,component=0,index=n_elements,value=0.2)])

# Example using the isotropic constitutive relation
element_batches_iso = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=1,
        connectivity_en=cells,
        constitutive_model=elastic_isotropic,
        material_params=matrix_mat_params,
    )
]

u_iso, residual_iso, element_batches_iso = solve_bvp(
    element_residual_func=linear_elasticity_residual,
    vertices_vd=points,
    element_batches=element_batches_iso,
    boundary_conditions=bcs,
    solver_options=SolverOptions(
        linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        # linear_precond_type=PreconditionerType.JACOBI,
        # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
        nonlinear_max_iter=1,
        linear_max_iter=5,
    ),
    plot_convergence=True,
)
print("\n*** Isotropic constitutive_model ***")
print("|R| = ", jnp.linalg.norm(residual_iso))
print(f"u = {u_iso}")
dirichlet_dofs = np.array([bc.index for bc in bcs])
dirichlet_values = np.array([bc.value for bc in bcs])
assert jnp.isclose(u_iso[dirichlet_dofs], dirichlet_values).all(), f"Dirichlet is not satisfied"

# Example using the truss elements
element_batches_truss = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=1,
        connectivity_en=cells,
        constitutive_model=elastic_truss,
        material_params=matrix_mat_params,
    )
]

u_truss, residual_truss, element_batches_truss = solve_bvp(
    element_residual_func=linear_truss_residual,
    vertices_vd=points,
    element_batches=element_batches_truss,
    boundary_conditions=bcs,
    solver_options=SolverOptions(
        linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        # linear_precond_type=PreconditionerType.JACOBI,
        # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
        nonlinear_max_iter=1,
        linear_max_iter=5,
    ),
    plot_convergence=True,
)
print("\n*** Truss Eleemnts! ***")
print("|R| = ", jnp.linalg.norm(residual_truss))
print(f"u = {u_truss}")
dirichlet_dofs = np.array([bc.index for bc in bcs])
dirichlet_values = np.array([bc.value for bc in bcs])
assert jnp.isclose(u_truss[dirichlet_dofs], dirichlet_values).all(), f"Dirichlet is not satisfied"

#Check that the two solutions match! 
assert jnp.isclose(u_truss,u_iso).all(), "The solutions from the isotropic model and the truss model do NOT match!"
print("The solutions from the isotropic and truss models match (at least to JAX default precision)!")
