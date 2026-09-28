from helper import *
import pytest
import time
import jetsci
import pyvista as pv

pytestmark = pytest.mark.slow


def run_bvp(mesh_path):
    jax.config.update("jax_enable_x64", True)

    args = {}
    args['t_total']  = 500
    import os
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    args['dir_path'] = f"nonlinear_IGFEM_vmap_t{args['t_total']}_{base_name}_JetSCI"
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # Read in the mesh
    mesh   = pv.read(mesh_path)
    points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
    cells = np.array(mesh.cells.data, dtype=np.uint64)
    dofs = 2 * points.shape[0]
    import re
    match = re.search(r'(\d+)fib', base_name)
    num_fibers = match.group(1) if match else "Unknown"
    print("# DoFs = ", dofs)
    print(f"# Fibers = {num_fibers}")

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

    num_cells = len(mesh.celltypes)
    matrix_tri_cells, matrix_quad_cells = [],[]
    matrix_tri_id, matrix_quad_id = [],[]
    fiber_tri_cells, fiber_quad_cells = [],[]
    fiber_tri_id, fiber_quad_id = [],[]

    matrix_ID = jnp.max(mesh.cell_data['materials'])
    for id, celltype in enumerate(mesh.celltypes):
        materials_ID = mesh.cell_data['materials'][id]    # Fiber=0, matrix= largest in the 'material'
        cell_nodes = mesh.get_cell(id).point_ids
        cell_nodes = reorder_cell_basix(mesh, cell_nodes)
        # Triangle
        if celltype == 5:
            if materials_ID == matrix_ID:
                matrix_tri_cells.append(cell_nodes)
                matrix_tri_id.append(id)
            else:
                fiber_tri_cells.append(cell_nodes)
                fiber_tri_id.append(id)
        # Quad
        elif celltype == 9:
            if materials_ID == matrix_ID:
                matrix_quad_cells.append(cell_nodes)
                matrix_quad_id.append(id)
            else:
                fiber_quad_cells.append(cell_nodes)
                fiber_quad_id.append(id)

    matrix_tri_cells,  matrix_quad_cells = np.array(matrix_tri_cells), np.array(matrix_quad_cells)
    matrix_tri_id,     matrix_quad_id    = np.array(matrix_tri_id),    np.array(matrix_quad_id)
    fiber_tri_cells,   fiber_quad_cells  = np.array(fiber_tri_cells),  np.array(fiber_quad_cells)
    fiber_tri_id,      fiber_quad_id     = np.array(fiber_tri_id),     np.array(fiber_quad_id)

    length = (np.max(mesh.points[:,0]) - np.min(mesh.points[:,0]))

    # Sizes of arrays
    U = 2  # number of solution components
    V = points.shape[0]  # number of vertices
    E = cells.shape[0]  # number of elements
    M = 11  # number of material parameters
    F = V * U  # number of DoFs
    Q_tri = 1 #get_quadrature(fe_type=fe_type_tri)[0].shape[0] # number of quadrature points
    Q_quad = 4 #get_quadrature(fe_type=fe_type_quad)[0].shape[0] # number of quadrature points

    strain_increment = args['strain_max'] * length / args['t_total']

    fe_type_tri = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=1,
    )
    fe_type_quad = FiniteElementType(
        cell_type=CellType.quadrilateral,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    # Define Dirichlet boundary conditions
    '''
    this BC locks the dx AND dy on both lhs and rhs 
    '''
    # LHS and RHS
    LHS = np.where(points[:,0]==np.min(points[:,0]))[0]
    RHS = np.where(points[:,0]==np.max(points[:,0]))[0]
    boundary_points = np.concatenate((LHS,RHS))
    dirichlet_values_init = np.zeros((boundary_points.shape[0], 2))
    n_LHS = len(LHS)
    n_vals = dirichlet_values_init.shape[0]
    dirichlet_values = np.empty(2 * n_vals)
    dirichlet_values[:n_vals] = dirichlet_values_init[:, 0]
    dirichlet_values[n_vals:] = dirichlet_values_init[:, 1]
    # An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
    # Constrain every boundary point to have a random displacement
    dirichlet_bcs = [
        DirichletBC(index=i, component=j, value=1.0)
        for j in range(U)
        for i in boundary_points
    ]

    '''    
    # An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
    dirichlet_bcs = np.zeros((U * boundary_points.shape[0], 2), dtype=np.uint64)
    for i,bp in enumerate(boundary_points):
        for j in range(U):
            dirichlet_bcs[U * i + j, 0] = bp
            dirichlet_bcs[U * i + j, 1] = j
    # Values of the Dirichlet boundary conditions matching 'dirichlet_bcs'
    dirichlet_values_init = np.zeros((boundary_points.shape[0], 2))
    '''    

    # Set material properties
    @partial(jax.jit, static_argnames=('Q',))
    def get_properties(matrix_cells,fiber_cells,Q: int):
        # Neat 5220 Epoxy
        matrix_mat_params_eqm = jnp.zeros(shape=(matrix_cells.shape[0], Q, 11))
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(3900)     # E   MPa
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 1].set(0.39)     # nu
        # Epoxy damage properties
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 2].set(0.95)     # P1, A
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 3].set(2.0)      # P2, B
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 4].set(10.0)     # mu_visc
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 5].set(0.35)     # eps_c
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 6].set(0.04)     # eps_t
        # Epoxy hardening properties
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 7].set(79)       # sigmay_c  MPa
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 8].set(62)       # sigmay_t  MPa
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 9].set(2e4)      # H_ro, a
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 10].set(12)      # n_ro, b 
        # IM7 Fiber
        fiber_mat_params_eqm = jnp.zeros(shape=(fiber_cells.shape[0], Q, 5))
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(233e3)  # E_xx
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(23.1e3) # E_yy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(0.2)    # nu_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(8.96e3) # G_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 4].set(8.27e3) # G_yz

        return (matrix_mat_params_eqm, fiber_mat_params_eqm)

    matrix_tri_mat_params_eqm, fiber_tri_mat_params_eqm   = get_properties(matrix_tri_cells,fiber_tri_cells,Q_tri)
    matrix_quad_mat_params_eqm, fiber_quad_mat_params_eqm = get_properties(matrix_quad_cells,fiber_quad_cells,Q_quad)

    # Intialize internal_state_eqi
    @partial(jax.jit, static_argnames=('Q',))
    def init_ISV(matrix_cells,fiber_cells,Q):
        # Intialize internal_state_eqi
        matrix_internal_state_eqi = jnp.zeros(shape=(matrix_cells.shape[0], Q, 11))
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,0:3].set(0)    # e11, e22, e12
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,3:6].set(0)    # s11, s22, s12
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,6]  .set(0)    # D
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,7]  .set(0)    # Y
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,8]  .set(0)    # tau0
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,9]  .set(0)    # vM0
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,10] .set(dt)   # dt
        
        fiber_internal_state_eqi = jnp.zeros(shape=(fiber_cells.shape[0], Q, 7))
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,0:3].set(0)     # e11
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,3:6].set(0)     # e22
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,6]  .set(-0.2)  # D
        
        return [matrix_internal_state_eqi, fiber_internal_state_eqi]

    internal_state_tri_eqi  = init_ISV(matrix_tri_cells,fiber_tri_cells,Q_tri)
    internal_state_quad_eqi = init_ISV(matrix_quad_cells,fiber_quad_cells,Q_quad)

    element_batches = [
        ElementBatch(
            fe_type=fe_type_tri,
            connectivity_en=matrix_tri_cells,
            constitutive_model=damage_elastic_isotropic_vmap,
            material_params=matrix_tri_mat_params_eqm,
            internal_state=internal_state_tri_eqi[0],
            n_dofs_per_basis=2,
        ),
        ElementBatch(
            fe_type=fe_type_quad,
            connectivity_en=matrix_quad_cells,
            constitutive_model=damage_elastic_isotropic_vmap,
            material_params=matrix_quad_mat_params_eqm,
            internal_state=internal_state_quad_eqi[0],
            n_dofs_per_basis=2,
        ),
        ElementBatch(
            fe_type=fe_type_tri,
            connectivity_en=fiber_tri_cells,
            constitutive_model=elastic_orthotropic,
            material_params=fiber_tri_mat_params_eqm,
            internal_state=internal_state_tri_eqi[1],
            n_dofs_per_basis=2,
        ),
        ElementBatch(
            fe_type=fe_type_quad,
            connectivity_en=fiber_quad_cells,
            constitutive_model=elastic_orthotropic,
            material_params=fiber_quad_mat_params_eqm,
            internal_state=internal_state_quad_eqi[1],
            n_dofs_per_basis=2,
        )
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

    petsc_solver = build_bvp_solver(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=dirichlet_bcs,
        multipoint_constraints=None,
        u_0_g=u_0,
        options=UnifiedBVPOptions(
            backend=BVPBackend.PETSC,
            petsc_solver_options=jetsci.SolverOptions(
                nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
                linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
                linear_solve_type=jetsci.PETScLinearSolverType.CG,
                nonlinear_absolute_tol=1e-14,
                linear_max_iter=10000,
                linear_relative_tol=1e-6,
                linear_absolute_tol=1e-14,
            ),
        ),
    )
    jax_solver = build_bvp_solver(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=dirichlet_bcs,
        multipoint_constraints=None,
        u_0_g=u_0,
        options=UnifiedBVPOptions(
            backend=BVPBackend.JAX,
            jax_solver_options=SolverOptions(
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                linear_precond_type=PreconditionerType.JACOBI,
            ),
        ),
    )

    try:
        petsc_result, _, petsc_times = time_solve("PETSc UnifiedBVPSolver", petsc_solver.solve, n_calls=3)
    finally:
        petsc_solver.destroy()

    jax_result, _, jax_times = time_solve("JAX UnifiedBVPSolver", jax_solver.solve, n_calls=3)
    u, residual, element_batches = jax_result
    u_petsc, residual_petsc, _ = petsc_result

    print("|R| PETSc = ", jnp.linalg.norm(residual_petsc))
    print("|R| JAX   = ", jnp.linalg.norm(residual))
    print("|u_petsc - u_jax| = ", jnp.linalg.norm(u_petsc - u))

    petsc_avg = sum(petsc_times) / len(petsc_times)
    petsc_min = min(petsc_times)
    petsc_max = max(petsc_times)
    
    jax_avg = sum(jax_times) / len(jax_times)
    jax_min = min(jax_times)
    jax_max = max(jax_times)
    
    return dofs, petsc_avg, petsc_min, petsc_max, jax_avg, jax_min, jax_max

def test_microscale_bvp(plot=False):
    meshes = [
        # "tests/meshes/IGFEM_1fib.vtk",
        # "tests/meshes/IGFEM_2fib.vtk",
        # "tests/meshes/IGFEM_4fib.vtk",
        # "tests/meshes/IGFEM_9fib_struct.vtk",
        # "tests/meshes/IGFEM_9fib.vtk",
        # "tests/meshes/IGFEM_16fib.vtk",
        # "tests/meshes/IGFEM_23fib.vtk",
        # "tests/meshes/IGFEM_36fib.vtk",
        # "tests/meshes/IGFEM_46fib.vtk",
        "tests/meshes/IGFEM_49fib.vtk",
    ]

    import matplotlib.pyplot as plt

    dofs_list = []
    petsc_avg_list = []
    petsc_min_list = []
    petsc_max_list = []
    jax_avg_list = []
    jax_min_list = []
    jax_max_list = []

    for mesh_path in meshes:
        print(f"--- Running mesh: {mesh_path} ---")
        try:
            dofs, p_avg, p_min, p_max, j_avg, j_min, j_max = run_bvp(mesh_path)
            dofs_list.append(dofs)
            petsc_avg_list.append(p_avg)
            petsc_min_list.append(p_min)
            petsc_max_list.append(p_max)
            jax_avg_list.append(j_avg)
            jax_min_list.append(j_min)
            jax_max_list.append(j_max)
        except Exception as e:
            print(f"Failed on {mesh_path}: {e}")

    # Plotting
    if plot:
        # Sort by DOFs for plotting
        sorted_indices = np.argsort(dofs_list)
        dofs_list = np.array(dofs_list)[sorted_indices]
        petsc_avg_list = np.array(petsc_avg_list)[sorted_indices]
        petsc_min_list = np.array(petsc_min_list)[sorted_indices]
        petsc_max_list = np.array(petsc_max_list)[sorted_indices]
        jax_avg_list = np.array(jax_avg_list)[sorted_indices]
        jax_min_list = np.array(jax_min_list)[sorted_indices]
        jax_max_list = np.array(jax_max_list)[sorted_indices]
    
        plt.rcParams['font.size'] = 14
        plt.figure(figsize=(8, 6))
    
        # Error bars for min/max
        p_err_lower = petsc_avg_list - petsc_min_list
        p_err_upper = petsc_max_list - petsc_avg_list
        j_err_lower = jax_avg_list - jax_min_list
        j_err_upper = jax_max_list - jax_avg_list
    
        plt.errorbar(dofs_list, petsc_avg_list, yerr=[p_err_lower, p_err_upper], fmt='o-', color='#4DA6FF', linewidth=2.5, markersize=8, label='PETSc (JetSCI)', capsize=5)
        plt.errorbar(dofs_list, jax_avg_list, yerr=[j_err_lower, j_err_upper], fmt='s-', color='#2CA02C', linewidth=2.5, markersize=8, label='JAX', capsize=5)
    
        plt.xlabel('Degrees of Freedom (DOFs)')
        plt.ylabel('Time (seconds)')
        plt.title('Computation Time vs. Degrees of Freedom (DOFs)')
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('dofs_vs_time.png', dpi=300)
        print("Saved plot to dofs_vs_time.png")

if __name__ == "__main__":
    test_microscale_bvp()
