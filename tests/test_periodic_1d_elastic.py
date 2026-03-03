from helper import *

jax.config.update("jax_enable_x64", True)

import jax.extend

print(jax.extend.backend.get_backend().platform)

# from jax_smi import initialise_tracking
# initialise_tracking()

n_elements = 3
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
element_batches = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=1,
        connectivity_en=cells,
        constitutive_model=elastic_isotropic,
        material_params=matrix_mat_params,
    )
]

# Sets of boundary conditions to test
mpcs = [
    [
        MultiPointConstraint(
            dep_dof=F - 1, indep_dofs=[0], factors=[1.0], rhs_constant=0.1
        )
    ],  # Case 1
    [
        MultiPointConstraint(
            dep_dof=F - 1, indep_dofs=[0], factors=[1.0], rhs_constant=0.1
        )
    ],  # Case 2
    [
        MultiPointConstraint(
            dep_dof=F - 1, indep_dofs=[0], factors=[1.0], rhs_constant=0.1
        )
    ],  # Case 3
]
dirichlet_constraints = [
    [
        DirichletConstraint(dep_dof=1, value=0.05),
    ],  # Case 1
    [
        DirichletConstraint(dep_dof=1, value=0.05),
        DirichletConstraint(dep_dof=2, value=0.1),
    ],  # Case 2
    [
        DirichletConstraint(dep_dof=0, value=0.05),
    ],  # Case 3
]


for i, (mc, dc) in enumerate(zip(mpcs, dirichlet_constraints)):
    print(f"\n************\nCase {i}\n")
    print(f"Multipoint constraints:\n{mc}")
    print(f"Dirichlet constraints:\n{dc}")

    # Solve the boundary value problem
    u, residual, element_batches = solve_bvp(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        dirichlet_bcs=dc,
        multipoint_constraints=mc,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=1,
            linear_max_iter=5,
        ),
        plot_convergence=True,
    )
    print("|R| = ", jnp.linalg.norm(residual))
    print(f"u = {u}")

    # Ensure multipoint constraints are satisified
    for c in mc:
        assert (
            abs(
                u[c.dep_dof]
                - sum([factor * u[dof] for dof, factor in c.indep_dof_terms.items()])
                - c.rhs_constant
            )
            < 1e-6
        ), f"MPC {c} is not satisfied"

    # Ensure Dirichlet BCs are satisfied
    for c in dc:
        assert abs(u[c.dep_dof] - c.value) < 1e-6, f"Dirichlet {c} is not satisfied"
