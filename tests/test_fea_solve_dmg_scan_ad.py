import meshio
import numpy as np

import fea_traditional as test
from helper import *

import os
import pyvista as pv
import time
import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.post_pred import *
from fe_jax.write_vtk import *
from fe_jax.linear_elasticity_dmg import *
from fe_jax.fea import convert_boundary_conditions, solve_bvp_quasi, SolverOptions, LinearSolverType

def test_ad():
    args = {}
    num_fib = '1'
    args['t_total']  = 500
    args['dir_path'] = "debug_opt"
    # args['dir_path'] = f"nonlinear_IGFEM_scan_t{args['t_total']}_{num_fib}fib_AD"
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # initial_params = jnp.array([233e3, 23.1e3, 0.2, 8.96e3, 8.27e3])
    # params = jnp.array([3900.0, 233.e3])    #opt
    # params = jnp.array([2000.0, 200.e3])    #1 
    # params = jnp.array([2000.0, 250.e3])    #2
    params = jnp.array([3000.0, 200.e3])    #3
    # params = jnp.array([4000.0, 250.e3])    #4
    # params = jnp.array([4250.0, 250.e3])    #5
    print("Initial parameters:", params)

    # Create the directory (and any necessary parent directories)
    args['out_dir'] = os.path.join("tests/output",args['dir_path'])
    args['vtk_dir'] = os.path.join(args['out_dir'],"vtks")
    os.makedirs(args['out_dir'], exist_ok=True)
    os.makedirs(args['vtk_dir'], exist_ok=True)

    # Read in the mesh (IGFEM)
    mesh = pv.read(f'tests/meshes/IGFEM_{num_fib}fib.vtk')
    vtk_mesh = mesh.copy()
    vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{0}.vtk")
    points = np.array(mesh.points, dtype=np.float64)[:,0:2]
    cells = np.array(mesh.cells, dtype=np.uint64)
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

    length = (np.max(mesh.points[:,0]) - np.min(mesh.points[:,0]))
    U = 2

    Q_tri = 1 #get_quadrature(fe_type=fe_type_tri)[0].shape[0] # number of quadrature points
    Q_quad = 4 #get_quadrature(fe_type=fe_type_quad)[0].shape[0] # number of quadrature points

    dx_increment = args['strain_max'] * length / args['t_total']

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
    # 1. Create an array of strains for every time step: [strain*1, strain*2, ..., strain*t_total]
    steps = jnp.arange(1, args['t_total'] + 1, dtype=jnp.float64)
    dxs = dx_increment * steps

    # LHS and RHS
    LHS = np.where(points[:,0]==np.min(points[:,0]))[0]
    RHS = np.where(points[:,0]==np.max(points[:,0]))[0]
    boundary_points = np.concatenate((LHS,RHS))
    
    # An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
    dirichlet_bcs = []
    for j in range(U):
        for i in boundary_points:
            # Set value to 0.0 by default
            val = 0.0
            # If it's x-displacement (j=0) and on the RHS, use time-varying dxs
            if j == 0 and i in RHS:
                val = dxs
            dirichlet_bcs.append(DirichletBC(index=i, component=j, value=val))

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
    def get_properties(matrix_cells,fiber_cells,Q: int, opt_params):
        # Neat 5220 Epoxy
        matrix_mat_params_eqm = jnp.zeros(shape=(matrix_cells.shape[0], Q, 11))
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(opt_params[0])     # E   MPa
        # matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(3900)     # E   MPa
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
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(opt_params[1])      # E_xx      MPa (233 GPa)
        # fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(233e3)      # E_xx      MPa (233 GPa)
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(23.1e3)     # E_yy      MPa (23.1 GPa)
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(0.2)        # nu_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(8.96e3)     # G_xy      MPa (8.96 GPa)
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 4].set(8.27e3)     # G_yz      MPa (8.27 GPa)

        return (matrix_mat_params_eqm, fiber_mat_params_eqm)

    matrix_tri_mat_params_init, fiber_tri_mat_params_init = get_properties(matrix_tri_cells, fiber_tri_cells, Q_tri, params)
    matrix_quad_mat_params_init, fiber_quad_mat_params_init = get_properties(matrix_quad_cells, fiber_quad_cells, Q_quad, params)

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
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,6]  .set(0)     # D
        
        return [matrix_internal_state_eqi, fiber_internal_state_eqi]

    internal_state_tri_m, internal_state_tri_f = init_ISV(matrix_tri_cells, fiber_tri_cells, Q_tri)
    internal_state_quad_m, internal_state_quad_f = init_ISV(matrix_quad_cells, fiber_quad_cells, Q_quad)


    # Auto-detect the index of the optimized parameter for matrix elements
    # based on where params[0] and params[1] are placed in the property array
    _valid_matrix_params = matrix_tri_mat_params_init if matrix_tri_mat_params_init.shape[0] > 0 else matrix_quad_mat_params_init
    opt_idx_0 = int(jnp.where(_valid_matrix_params[0, 0] == params[0])[0][0])

    _valid_fiber_params = fiber_tri_mat_params_init if fiber_tri_mat_params_init.shape[0] > 0 else fiber_quad_mat_params_init
    opt_idx_1 = int(jnp.where(_valid_fiber_params[0, 0] == params[1])[0][0])

    # Precompute indices for the optimized parameters
    S_1 = matrix_tri_cells.shape[0] * Q_tri * 11
    S_2 = matrix_quad_cells.shape[0] * Q_quad * 11
    S_3 = fiber_tri_cells.shape[0] * Q_tri * 5
    S_4 = fiber_quad_cells.shape[0] * Q_quad * 5

    opt_param_indices_0 = jnp.concatenate([
        jnp.arange(opt_idx_0, S_1, 11),
        S_1 + jnp.arange(opt_idx_0, S_2, 11)
    ])
    opt_param_indices_1 = jnp.concatenate([
        (S_1 + S_2) + jnp.arange(opt_idx_1, S_3, 5),
        (S_1 + S_2 + S_3) + jnp.arange(opt_idx_1, S_4, 5)
    ])

    element_batches = [
        ElementBatch(
            fe_type=fe_type_tri,
            connectivity_en=matrix_tri_cells,
            constitutive_model=damage_elastic_isotropic_vmap,
            material_params=matrix_tri_mat_params_init,
            internal_state=internal_state_tri_m,
            n_dofs_per_basis=2,
        ),
        ElementBatch(
            fe_type=fe_type_quad,
            connectivity_en=matrix_quad_cells,
            constitutive_model=damage_elastic_isotropic_vmap,
            material_params=matrix_quad_mat_params_init,
            internal_state=internal_state_quad_m,
            n_dofs_per_basis=2,
        ),
        ElementBatch(
            fe_type=fe_type_tri,
            connectivity_en=fiber_tri_cells,
            constitutive_model=elastic_orthotropic,
            material_params=fiber_tri_mat_params_init,
            internal_state=internal_state_tri_f,
            n_dofs_per_basis=2,
        ),
        ElementBatch(
            fe_type=fe_type_quad,
            connectivity_en=fiber_quad_cells,
            constitutive_model=elastic_orthotropic,
            material_params=fiber_quad_mat_params_init,
            internal_state=internal_state_quad_f,
            n_dofs_per_basis=2,
        )
    ]

    print('Start Deformation Loop')

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        boundary_conditions=dirichlet_bcs,
    )

    solve_nonlinear_step_jit = jax.jit(
        solve_nonlinear_quasi_step_autodiff_vjp,
        static_argnames=["solver_options", "jacobian_nnz"],
    )

    n_total_dofs = points.shape[0] * ebc.U[0]

    all_g = constraint_system.g
    if all_g.ndim == 1:
        all_g = all_g[None, :]
        
    all_loads = f_ext.loads
    if all_loads.ndim == 1:
        all_loads = jnp.broadcast_to(all_loads[None, :], (all_g.shape[0], all_loads.shape[0]))

    # The forward pass which computes objective
    def forward(fiber_params):
        # matrix_tri_cells, matrix_quad_cells, fiber_tri_cells, fiber_quad_cells 
        new_material_params = ebc.material_params.at[opt_param_indices_0].set(fiber_params[0])
        new_material_params = new_material_params.at[opt_param_indices_1].set(fiber_params[1])
        
        ebc_updated = ebc.replace(material_params=new_material_params)
        u_prev = jnp.zeros(shape=(n_total_dofs,))

        def step_fn(carry, xs):
            u_prev_step, ebc_curr = carry
            g_i, loads_i = xs
            constraint_system_i = constraint_system.replace(g=g_i)
            f_ext_i = f_ext.replace(loads=loads_i)
            
            u, new_internal_state_beqi, residual, relative_error, info = solve_nonlinear_step_jit(
                element_residual_func=element_residual_func,
                ebc=ebc_curr,
                assembly_map_b=assembly_map_b,
                jacobian_nnz=jacobian_nnz,
                u_0_g=u_prev_step,
                constraints=constraint_system_i,
                solver_options=SolverOptions(
                    linear_solve_type=LinearSolverType.DENSE_INVERSE_JNP,
                ),
                f_ext=f_ext_i,
            )

            flat_internal_state = jnp.hstack([isv.ravel() for isv in new_internal_state_beqi])
            ebc_next = ebc_curr.replace(internal_state=flat_internal_state)
            ISV_be = compute_ISV_be(new_internal_state_beqi)

            return (u, ebc_next), (u, ISV_be)

        init_carry = (u_prev, ebc_updated)
        final_carry, (u_history, ISV_be_history) = jax.lax.scan(step_fn, init_carry, (all_g, all_loads))
        # s11_global = compute_stress_strain_curve(ISV_be_history, element_batches, points)
        s11_global, s22_global, s12_global = compute_stress_strain_curve(ISV_be_history, element_batches, points)

        obj = -jnp.max(s11_global)
        # loss = jnp.mean((s_match - jnp.stack([s11_global, s22_global, s12_global], axis=1)) ** 2)

        return obj, (ISV_be_history, s11_global, u_history)

    print("Evaluating forward pass to compile...")
    start_t = time.time()
    obj, _ = forward(params)
    print("Objective:", obj)
    print("Forward compiled and executed in", time.time() - start_t, "seconds")
    
    print("Compiling jax.value_and_grad...")
    start_t = time.time()
    val_grad_fn = jax.jit(jax.value_and_grad(forward, has_aux=True))
    (loss, _), grad = val_grad_fn(params)
    print(f"Initial compiled evaluation - Loss: {loss}, Grad: {grad}")
    print("Grad compiled and executed in", time.time() - start_t, "seconds")

    print("\n--- Starting optimization loop ---")
    
    lr = jnp.array([1e4, 1e7])
    
    def opt_step(carry, i):
        curr_phys_params = carry
        # curr_phys_params = curr_phys_params.at[0].set( 233.e3 + 3900. - curr_phys_params[1])
        (loss, _), phys_grad = val_grad_fn(curr_phys_params)
        
        jax.debug.print("Iteration {i} | Loss: {loss} | Phys Params: {params}| sum Params: {sum}", 
                        i=i+1, loss=loss, params=curr_phys_params, sum=jnp.sum(curr_phys_params))
                        
        next_phys_params = curr_phys_params - lr * phys_grad
        
        # Project parameters into physically realistic bounds [Epoxy E, Fiber E_xx]
        lower_bounds = jnp.array([2500.0, 230000.0])
        upper_bounds = jnp.array([4500.0, 26000.0])
        next_phys_params = jnp.clip(next_phys_params, min=lower_bounds, max=upper_bounds)
        
        return next_phys_params, (curr_phys_params, loss)
        
    @jax.jit
    def run_optimization(init_phys_params):
        final_carry, (param_hist, loss_hist) = jax.lax.scan(
            opt_step, 
            init_phys_params, 
            jnp.arange(50)
        )
        return final_carry, param_hist, loss_hist

    params, param_hist, loss_hist = run_optimization(params)
    
    print("Final parameters:", params)
    
    print("Running final forward pass to get optimized histories...")
    final_loss, (final_ISV_history, final_s11, final_u_history) = forward(params)
    
    # Save outputs
    np.save(os.path.join(args['out_dir'], "param_hist.npy"), param_hist)
    np.save(os.path.join(args['out_dir'], "loss_hist.npy"), loss_hist)
    # np.save(os.path.join(args['out_dir'], "final_ISV_history.npy"), final_ISV_history)
    np.save(os.path.join(args['out_dir'], "final_s11_global.npy"), final_s11)
    # np.save(os.path.join(args['out_dir'], "final_u_history.npy"), final_u_history)

    return n_total_dofs, args['out_dir']

if __name__ == "__main__":
    t_start = time.time()

    n_total_dofs, out_dir = test_ad()
    total_time = time.time() - t_start
    print("Total time:", total_time)

    with open(os.path.join(out_dir, "statistics.txt"), "w") as f:
        f.write(f"Number of dofs: {n_total_dofs}\n")
        f.write(f"Total Solver time: {total_time} seconds\n")