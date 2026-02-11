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

# Periodic boundary conditions
mpcs = [
    MultiPointConstraint(dep_dof=F - 1, indep_dofs=[0], factors=[1.0], rhs_constant=0.1)
]

dirichlet_constraints = [
    DirichletConstraint(dep_dof=1, value=0.05),
    # DirichletConstraint(dep_dof=2, value=0.1),
]

print(mpcs)


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

ebc, assembly_map_b, constraint_system, jacobian_nnz, element_residual_func = (
    preprocess_bvp(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        dirichlet_bcs=dirichlet_constraints,
        multipoint_constraints=mpcs,
    )
)

# Check the Jacobian without any constraints

J = jax.jacfwd(
    lambda u: calculate_residual_wo_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u,
    ),
    argnums=0,
)(jnp.zeros(F))[0]

# Check the result against the analytical solution
assert jnp.isclose(
    J,
    jnp.array(
        [
            [3e9, -3e9, 0, 0],
            [-3e9, 6e9, -3e9, 0],
            [0, -3e9, 6e9, -3e9],
            [0, 0, -3e9, 3e9],
        ]
    ),
).all()

# Check the Jacobian with constraints

J_constrained = jax.jacfwd(
    lambda u: calculate_residual_w_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u,
        constraints=constraint_system,
    ),
    argnums=0,
)(jnp.zeros(F))[0]

print("J_constrained = \n", J_constrained)


P_full = (
    jnp.diag(jnp.ones(F))
    .at[constraint_system.dep_dofs]
    .set(constraint_system.P.todense())
)
print(P_full)

J_constrained_reference = (
    (P_full.T @ J @ P_full)
    .at[constraint_system.dep_dofs, constraint_system.dep_dofs]
    .set(1.0)
)
print("J_constrained_reference = \n", J_constrained_reference)

assert jnp.isclose(J_constrained, J_constrained_reference).all()
