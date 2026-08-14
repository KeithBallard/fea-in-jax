from helper import *
import pytest
import time

pytestmark = pytest.mark.slow


def test_microscale_bvp():
    jax.config.update("jax_enable_x64", True)

    # import jax.extend
    # print(jax.extend.backend.get_backend().platform)

    # from jax_smi import initialise_tracking
    # initialise_tracking()

    # Read in the mesh
    mesh = meshio.read(get_mesh("microscale_2D_r0.vtk"))
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
    right_u_val = (max_xy[0] - min_xy[0]) / 100.0
    bcs = (
        [
            DirichletBC(bc_type=BCType.NODE, index=i, component=0, value=0.0)
            for i in left_points
        ]
        + [
            DirichletBC(
                bc_type=BCType.NODE, index=i, component=0, value=right_u_val
            )
            for i in right_points
        ]
        + [
            DirichletBC(bc_type=BCType.NODE, index=i, component=1, value=0.0)
            for i in bottom_points
        ]
        + [
            DirichletBC(bc_type=BCType.NODE, index=i, component=1, value=0.0)
            for i in top_points
        ]
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

    def make_element_batches():
        return [
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=2,
                connectivity_en=matrix_cells,
                constitutive_model=elastic_isotropic,
                material_params=matrix_mat_params_eqm,
            ),
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=2,
                connectivity_en=fiber_cells,
                constitutive_model=elastic_orthotropic,
                material_params=fiber_mat_params_eqm,
            ),
        ]

    u_0 = jnp.zeros(shape=(V * U))

    # Solve the boundary value problem

    def block_bvp_result(result):
        u, residual, element_batches = result
        u.block_until_ready()
        residual.block_until_ready()
        return u, residual, element_batches

    def time_solve(label, solve_func, n_calls=3):
        start = time.perf_counter()
        result = block_bvp_result(solve_func())
        first_call_time = time.perf_counter() - start

        times = []
        for _ in range(n_calls):
            start = time.perf_counter()
            result = block_bvp_result(solve_func())
            times.append(time.perf_counter() - start)

        print(
            f"{label}: first call {first_call_time:.6f} sec, "
            f"avg {sum(times) / len(times):.6f} sec "
            f"(n={n_calls}, min={min(times):.6f}, max={max(times):.6f})"
        )
        return result, first_call_time, times


    def run_petsc():
        return solve_bvp_PETSc(
            vertices_vd=points,
            element_batches=make_element_batches(),
            element_residual_func=linear_elasticity_residual,
            boundary_conditions=bcs,
            multipoint_constraints=None,
            u_0_g=u_0,
            diagnostics=False,
            petsc_solver_options=jetsci.SolverOptions(
                nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
                linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
                linear_solve_type=jetsci.PETScLinearSolverType.CG,
                nonlinear_absolute_tol=1e-14,
                linear_max_iter=10000,
                linear_relative_tol=1e-6,
                linear_absolute_tol=1e-14,
            ),
        )

    def run_jax():
        return solve_bvp(
            element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=make_element_batches(),
            u_0_g=u_0,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                linear_precond_type=PreconditionerType.JACOBI,
            ),
        )


    """
    petsc_result, _, _ = time_solve("PETSc solve_bvp_PETSc", run_petsc, n_calls=3)
    u_petsc, residual_petsc, _ = petsc_result
    print("|R| PETSc = ", jnp.linalg.norm(residual_petsc))

    exit(1)
    """

    """
    jax_result, _, _ = time_solve("JAX solve_bvp", run_jax, n_calls=2)
    u, residual, element_batches = jax_result
    print("|R| JAX   = ", jnp.linalg.norm(residual))










    print("|R| PETSc = ", jnp.linalg.norm(residual_petsc))
    print("|R| JAX   = ", jnp.linalg.norm(residual))
    print("|u_petsc - u_jax| = ", jnp.linalg.norm(u_petsc - u))

    # Make sure the solution matches at the Dirichlet BCs
    dirichlet_dofs = np.array([U * bc.index + bc.component for bc in bcs])
    dirichlet_values = np.array([bc.value for bc in bcs])
    assert jnp.isclose(u[dirichlet_dofs], dirichlet_values).all()

    # Write output
    mesh.point_data["u"] = u.reshape((points.shape[0], U))
    mesh.write(get_output("test_microscale_bvp_out.vtk"))
    """

test_microscale_bvp()
