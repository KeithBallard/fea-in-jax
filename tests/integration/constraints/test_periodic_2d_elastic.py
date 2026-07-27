from helper import *
import pytest

def test_periodic_2d_elastic_uniform_strain():
    jax.config.update("jax_enable_x64", True)

    nx, ny = 3, 3
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Generate cells (two triangles per quad)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i
            n3 = n2 + 1
            # Triangle 1
            cells.append([n0, n1, n2])
            # Triangle 2
            cells.append([n1, n3, n2])
    cells = np.array(cells, dtype=np.uint64)

    # Sizes of arrays
    U = 2  # number of solution components (u, v)
    V = points.shape[0]  # number of vertices
    E = cells.shape[0]  # number of elements

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    # Set material properties
    matrix_mat_params = jnp.array([1.0e9, 0.3])  # E, nu
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=cells,
            constitutive_model=elastic_isotropic,
            material_params=matrix_mat_params,
        )
    ]

    # Set up periodic boundary conditions
    bl_idx = 0
    br_idx = nx
    tl_idx = ny * (nx + 1)
    tr_idx = (ny + 1) * (nx + 1) - 1

    periodic_bcs = []
    # Corner pairs
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=br_idx, global_gradient_index=0))
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=tl_idx, global_gradient_index=0))
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=tr_idx, global_gradient_index=0))

    # Bottom/Top interior pairs
    for i in range(1, nx):
        bottom_idx = i
        top_idx = ny * (nx + 1) + i
        periodic_bcs.append(PeriodicBC(primary_index=bottom_idx, secondary_index=top_idx, global_gradient_index=0))

    # Left/Right interior pairs
    for j in range(1, ny):
        left_idx = j * (nx + 1)
        right_idx = j * (nx + 1) + nx
        periodic_bcs.append(PeriodicBC(primary_index=left_idx, secondary_index=right_idx, global_gradient_index=0))

    # Pin translation at BL corner
    pin_bcs = [
        DirichletBC(index=bl_idx, component=0, value=0.0),
        DirichletBC(index=bl_idx, component=1, value=0.0),
    ]

    # Prescribe macroscopic strain components
    eps_xx = 0.01
    eps_yy = -0.003
    eps_xy = 0.002

    prescribed_strain_bcs = [
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=0, value=eps_xx),
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=1, value=eps_yy),
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=2, value=eps_xy),
    ]

    boundary_conditions = periodic_bcs + pin_bcs + prescribed_strain_bcs

    # Solve BVP using CG solver (as we determined, CG uses JVP and correctly supports MPCs)
    u, residual, element_batches = solve_bvp(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=boundary_conditions,
        global_values=[3],  # size 3 block representing [eps_xx, eps_yy, eps_xy]
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            linear_max_iter=1000,
            linear_relative_tol=1e-12,
        ),
    )

    # Calculate analytical displacement field for comparison
    u_analytical = np.zeros_like(u)
    for idx in range(V):
        x_val, y_val = points[idx]
        u_analytical[2 * idx] = eps_xx * x_val + eps_xy * y_val
        u_analytical[2 * idx + 1] = eps_xy * x_val + eps_yy * y_val

    # Global DOFs (strain components at the end of the solution vector)
    u_analytical[2 * V] = eps_xx
    u_analytical[2 * V + 1] = eps_yy
    u_analytical[2 * V + 2] = eps_xy

    # Compare displacements
    print("Computed u:", u)
    print("Analytical u:", u_analytical)
    assert np.allclose(u, u_analytical, atol=1e-6)


def test_periodic_2d_elastic_direct_solver():
    jax.config.update("jax_enable_x64", True)

    nx, ny = 3, 3
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Generate cells
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i
            n3 = n2 + 1
            cells.append([n0, n1, n2])
            cells.append([n1, n3, n2])
    cells = np.array(cells, dtype=np.uint64)

    U = 2
    V = points.shape[0]

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    matrix_mat_params = jnp.array([1.0e9, 0.3])
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=cells,
            constitutive_model=elastic_isotropic,
            material_params=matrix_mat_params,
        )
    ]

    bl_idx = 0
    br_idx = nx
    tl_idx = ny * (nx + 1)
    tr_idx = (ny + 1) * (nx + 1) - 1

    periodic_bcs = []
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=br_idx, global_gradient_index=0))
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=tl_idx, global_gradient_index=0))
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=tr_idx, global_gradient_index=0))

    for i in range(1, nx):
        bottom_idx = i
        top_idx = ny * (nx + 1) + i
        periodic_bcs.append(PeriodicBC(primary_index=bottom_idx, secondary_index=top_idx, global_gradient_index=0))

    for j in range(1, ny):
        left_idx = j * (nx + 1)
        right_idx = j * (nx + 1) + nx
        periodic_bcs.append(PeriodicBC(primary_index=left_idx, secondary_index=right_idx, global_gradient_index=0))

    pin_bcs = [
        DirichletBC(index=bl_idx, component=0, value=0.0),
        DirichletBC(index=bl_idx, component=1, value=0.0),
    ]

    eps_xx = 0.01
    eps_yy = -0.003
    eps_xy = 0.002

    prescribed_strain_bcs = [
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=0, value=eps_xx),
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=1, value=eps_yy),
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=2, value=eps_xy),
    ]

    boundary_conditions = periodic_bcs + pin_bcs + prescribed_strain_bcs

    # Solve BVP using the direct sparse solver (which uses coo_to_csr and apply_mpc_to_jacobian)
    u, residual, element_batches = solve_bvp(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=boundary_conditions,
        global_values=[3],
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.SPSOLVE_CUPY,
        ),
    )

    u_analytical = np.zeros_like(u)
    for idx in range(V):
        x_val, y_val = points[idx]
        u_analytical[2 * idx] = eps_xx * x_val + eps_xy * y_val
        u_analytical[2 * idx + 1] = eps_xy * x_val + eps_yy * y_val

    u_analytical[2 * V] = eps_xx
    u_analytical[2 * V + 1] = eps_yy
    u_analytical[2 * V + 2] = eps_xy

    assert np.allclose(u, u_analytical, atol=1e-6)


def test_global_relation_bc():
    jax.config.update("jax_enable_x64", True)

    nx, ny = 3, 3
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Generate cells
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i
            n3 = n2 + 1
            cells.append([n0, n1, n2])
            cells.append([n1, n3, n2])
    cells = np.array(cells, dtype=np.uint64)

    U = 2

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    matrix_mat_params = jnp.array([1.0e9, 0.3])
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=cells,
            constitutive_model=elastic_isotropic,
            material_params=matrix_mat_params,
        )
    ]

    bl_idx = 0
    br_idx = nx
    tl_idx = ny * (nx + 1)
    tr_idx = (ny + 1) * (nx + 1) - 1

    periodic_bcs = []
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=br_idx, global_gradient_index=0))
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=tl_idx, global_gradient_index=0))
    periodic_bcs.append(PeriodicBC(primary_index=bl_idx, secondary_index=tr_idx, global_gradient_index=0))

    for i in range(1, nx):
        bottom_idx = i
        top_idx = ny * (nx + 1) + i
        periodic_bcs.append(PeriodicBC(primary_index=bottom_idx, secondary_index=top_idx, global_gradient_index=0))

    for j in range(1, ny):
        left_idx = j * (nx + 1)
        right_idx = j * (nx + 1) + nx
        periodic_bcs.append(PeriodicBC(primary_index=left_idx, secondary_index=right_idx, global_gradient_index=0))

    pin_bcs = [
        DirichletBC(index=bl_idx, component=0, value=0.0),
        DirichletBC(index=bl_idx, component=1, value=0.0),
    ]

    eps_xx = 0.01
    eps_yy = -0.003
    # Use GlobalRelationBC to tie component 2 (eps_xy) to twice component 0 (eps_xx)
    # eps_xy = 2.0 * eps_xx + 0.001
    relation_bc = GlobalRelationBC(index_secondary=0, component_secondary=2, index_primary=0, component_primary=0, factor=2.0, constant=0.001)

    prescribed_strain_bcs = [
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=0, value=eps_xx),
        DirichletBC(bc_type=BCType.GLOBAL_VALUE, index=0, component=1, value=eps_yy),
    ]

    boundary_conditions = periodic_bcs + pin_bcs + prescribed_strain_bcs + [relation_bc]

    u, residual, element_batches = solve_bvp(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=boundary_conditions,
        global_values=[3],
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        ),
    )

    # Check that eps_xy solved to 2.0 * eps_xx + 0.001
    computed_eps_xy = u[-1]
    expected_eps_xy = 2.0 * eps_xx + 0.001
    assert np.isclose(computed_eps_xy, expected_eps_xy, atol=1e-6)

