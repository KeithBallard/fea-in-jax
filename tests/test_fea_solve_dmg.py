import meshio
import numpy as np

import fea_traditional as test
from helper import *

import os
import pyvista as pv
import time
# jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.post_pred import *
from fe_jax.write_vtk import *
from fe_jax.linear_elasticity_dmg import *
from fe_jax.fea import convert_boundary_conditions
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)
 

def test_fea_solve_dmg():
    args = {}
    args['dir_path'] = "nonlinear_IGFEM_vmap_t500_4fib_keith"
    args['t_total']  = 500
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # Create the directory (and any necessary parent directories)
    args['out_dir'] = os.path.join("tests/output",args['dir_path'])
    args['vtk_dir'] = os.path.join(args['out_dir'],"vtks")
    os.makedirs(args['out_dir'], exist_ok=True)
    os.makedirs(args['vtk_dir'], exist_ok=True)

    # Read in the mesh (IGFEM)
    mesh     = pv.read('tests/meshes/IGFEM_4fib.vtk')
    vtk_mesh = mesh.copy()
    vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{0}.vtk")

    points = np.array(mesh.points, dtype=np.float64)[:,0:2]
    cells  = np.array(mesh.cells, dtype=np.uint64)
    print("# DoFs = ", 2 * points.shape[0])

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

    print_cell_ID = 12
    print_cell,fib_matrix_shape = find_print_cell_idx(mesh,print_cell_ID, matrix_tri_id,matrix_quad_id,fiber_tri_id,fiber_quad_id)

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

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
        solve_nonlinear_step_jit
    ) = preprocess_bvp(element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=element_batches,
            boundary_conditions=dirichlet_bcs)
    
    n_total_dofs = points.shape[0] * ebc.U[0]
    u_prev = jnp.zeros(shape=(n_total_dofs,))

    print('Start Deformation Loop')
    for i in range(1,args['t_total']+1):
        dirichlet_values[n_LHS:n_vals] = strain_increment * i
        # Update the constraint system's g vector directly to avoid slow CPU overhead 
        # of reconstructing the boundary conditions every iteration.
        constraint_system = constraint_system.replace(
            g=jnp.array(dirichlet_values, dtype=jnp.float32)
        )

        # Solve the boundary value problem
        u, internal_state_beqi, residual, relative_error, info = solve_nonlinear_step_jit(
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            jacobian_nnz=jacobian_nnz,
            u_0_g=u_prev,
            constraints=constraint_system,
            solver_options=SolverOptions(
                # linear_precond_type=PreconditionerType.JACOBI,
                # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
                # linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                linear_solve_type=LinearSolverType.DENSE_INVERSE_JNP,
            ),
            f_ext=f_ext,
        )

        u_prev = u

        # Update the internal state variables in the element batch collection for the next time step
        ebc = ebc.replace(internal_state=jnp.hstack([isv.ravel() for isv in internal_state_beqi]))

        print("Time step =", i)
        # Update displacements to be 3D for VTK on GPU, then transfer once to host
        u_full = np.array(jnp.zeros((points.shape[0], 3)).at[:, :U].set(u.reshape(-1, U)))

        # write and save to vtk
        vtk_mesh = write2VTK_avg(args,mesh,u_full,internal_state_beqi,fiber_tri_id,matrix_tri_id,fiber_quad_id,matrix_quad_id)
        # Write to file
        vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{i}.vtk")
    # zip_folder(args['vtk_dir'], args['vtk_dir']+'.zip')

if __name__ == "__main__":
    t_start = time.time()

    # with jax.profiler.trace("./jax-trace_new", create_perfetto_trace=True):
    test_fea_solve_dmg()

    t_end = time.time()
    print("Time used:", t_end - t_start)