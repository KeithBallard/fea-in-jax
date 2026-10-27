import meshio
import numpy as np

import fea_traditional as test
from helper import *
from igfem_mesh import *

import os
import pyvista as pv
import time
jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.post_pred import *
from fe_jax.write_vtk import *
from fe_jax.plot_results import plot_comparison
from fe_jax.linear_elasticity_dmg import *
from fe_jax.fea import convert_boundary_conditions
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)
 

def test_fea_solve_dmg():
    args = {}
    num_fib = 36    # 1, 2, 4, 9, 16, 23, 33, 36, 46, 49, 60, 64, 100
    # num_fib = int(os.environ.get("NUM_FIB", 1))    # 1, 2, 4, 9, 16, 23, 33, 36, 46, 49, 60, 64, 100
    args['t_total']  = 500
    # args['dir_path'] = "debug_update"
    args['dir_path'] = f"update_CG_t{args['t_total']}_{num_fib}fib"
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # Create the directory (and any necessary parent directories)
    args['out_dir'] = os.path.join("tests/output",args['dir_path'])
    args['vtk_dir'] = os.path.join(args['out_dir'],"vtks")
    args['ref_dir'] = os.path.join("tests/IGFEM_ref",f"{num_fib}fib_t{args['t_total']}")
    os.makedirs(args['out_dir'], exist_ok=True)
    os.makedirs(args['vtk_dir'], exist_ok=True)

    # Read in the mesh (IGFEM)
    mesh     = pv.read(f'tests/meshes/IGFEM_{num_fib}fib.vtk')
    vtk_mesh = mesh.copy()
    vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{0}.vtk")

    points = np.array(mesh.points, dtype=np.float64)[:,0:2]
    cells  = np.array(mesh.cells, dtype=np.uint64)
    print("# DoFs = ", 2 * points.shape[0])

    num_cells = len(mesh.celltypes)
    # {material: {shape: (cells, cell_ids)}} with only the shapes present in the mesh (tri, quad, polygon5, ...)
    cell_groups = sort_cells_by_material(mesh)

    length = (np.max(mesh.points[:,0]) - np.min(mesh.points[:,0]))

    # Sizes of arrays
    U = 2  # number of solution components
    V = points.shape[0]  # number of vertices
    E = cells.shape[0]  # number of elements
    M = 11  # number of material parameters
    F = V * U  # number of DoFs

    strain_increment = args['strain_max'] * length / args['t_total']

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
    def matrix_properties(cells, Q: int):
        # Neat 5220 Epoxy
        matrix_mat_params_eqm = jnp.zeros(shape=(cells.shape[0], Q, 11))
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
        return matrix_mat_params_eqm

    @partial(jax.jit, static_argnames=('Q',))
    def fiber_properties(cells, Q: int):
        # IM7 Fiber
        fiber_mat_params_eqm = jnp.zeros(shape=(cells.shape[0], Q, 5))
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(233e3)  # E_xx
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(23.1e3) # E_yy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(0.2)    # nu_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(8.96e3) # G_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 4].set(8.27e3) # G_yz
        return fiber_mat_params_eqm

    # Intialize internal_state_eqi
    @partial(jax.jit, static_argnames=('Q',))
    def matrix_ISV(cells, Q: int):
        matrix_internal_state_eqi = jnp.zeros(shape=(cells.shape[0], Q, 11))
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,0:3].set(0)    # e11, e22, e12
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,3:6].set(0)    # s11, s22, s12
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,6]  .set(0)    # D
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,7]  .set(0)    # Y
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,8]  .set(0)    # tau0
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,9]  .set(0)    # vM0
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,10] .set(dt)   # dt
        return matrix_internal_state_eqi

    @partial(jax.jit, static_argnames=('Q',))
    def fiber_ISV(cells, Q: int):
        fiber_internal_state_eqi = jnp.zeros(shape=(cells.shape[0], Q, 7))
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,0:3].set(0)     # e11
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,3:6].set(0)     # e22
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,6]  .set(-0.2)  # D
        return fiber_internal_state_eqi

    # (constitutive model, material parameters, initial ISV) of each material group
    materials = {
        "matrix": (damage_elastic_isotropic_vmap, matrix_properties, matrix_ISV),
        "fiber":  (elastic_orthotropic,           fiber_properties,  fiber_ISV),
    }

    # One element batch per (material, cell shape) present in the mesh
    element_batches = []
    batch_cell_ids = []     # mesh cell index of each element, per batch
    for name, shapes in cell_groups.items():
        constitutive_model, get_properties, init_ISV = materials[name]
        for shape, (cells_en, cell_ids) in shapes.items():
            fe_type = igfem_element_type(shape)
            Q = get_quadrature(fe_type=fe_type)[0].shape[0]    # number of quadrature points
            element_batches.append(ElementBatch(
                fe_type=fe_type,
                connectivity_en=cells_en,
                constitutive_model=constitutive_model,
                material_params=get_properties(cells_en, Q),
                internal_state=init_ISV(cells_en, Q),
                n_dofs_per_basis=2,
            ))
            batch_cell_ids.append(cell_ids)
            print(f"{name} {shape}: {cells_en.shape[0]} cells, Q = {Q}")

    # print_cell_ID = 12
    # print_cell,fib_matrix_shape = find_print_cell_idx(print_cell_ID, batch_cell_ids)

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=element_batches,
            boundary_conditions=dirichlet_bcs)
    
    solve_nonlinear_step_jit = jax.jit(
        solve_nonlinear_quasi_step,
        # donate_argnames="internal_state_beqi",
        static_argnames=["solver_options", "jacobian_nnz"],
    )

    n_total_dofs = points.shape[0] * ebc.U[0]
    u_prev = jnp.zeros(shape=(n_total_dofs,))

    # 1. Create an array of strains for every time step: [strain*1, strain*2, ..., strain*t_total]
    steps = jnp.arange(1, args['t_total'] + 1, dtype=jnp.float64)
    strains = strain_increment * steps
    
    # 2. Pre-allocate the `g` vector for ALL time steps. 
    # Shape will be: (t_total, size_of_dirichlet_values)
    all_g = jnp.tile(jnp.array(dirichlet_values, dtype=jnp.float64), (args['t_total'], 1))
    
    # 3. Apply the strain update for all time steps simultaneously.
    # strains[:, None] broadcasts the strain across all the nodes in the slice.
    all_g = all_g.at[:, n_LHS:n_vals].set(strains[:, None])

    print('Start Deformation Loop')
    for i in range(1,args['t_total']+1):
        # Update the constraint system's g vector directly to avoid slow CPU overhead of reconstructing the boundary conditions every iteration.
        constraint_system = constraint_system.replace(g=all_g[i - 1])
        
        # Solve the boundary value problem
        u, ISV_be, residual, relative_error, info = solve_nonlinear_step_jit(
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            jacobian_nnz=jacobian_nnz,
            u_0_g=u_prev,
            constraints=constraint_system,
            solver_options=SolverOptions(
                # linear_solve_type=LinearSolverType.DENSE_INVERSE_JNP,
                linear_precond_type=PreconditionerType.JACOBI,
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY,#_W_INFO,
                # linear_solve_type=LinearSolverType.SPSOLVE_CUPY, 
                linear_max_iter=10000,
            ),
            f_ext=f_ext,
        )
        u_prev = u

        # Update the internal state variables in the element batch collection for the next time step
        ebc = ebc.replace(internal_state=jnp.hstack([isv.ravel() for isv in ISV_be]))
        ISV_be1 = compute_ISV_be(ISV_be)
        print("Time step =", i)

        # write and save to vtk
        vtk_mesh = write2VTK_ISV(args,mesh,u,ISV_be1,batch_cell_ids)
        vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{i}.vtk")
    # zip_folder(args['vtk_dir'], args['vtk_dir']+'.zip')
    return n_total_dofs, args, vtk_mesh

if __name__ == "__main__":
    t_start = time.perf_counter()

    # with jax.profiler.trace("./jax-trace_new", create_perfetto_trace=True):
    n_total_dofs, args, vtk_mesh = test_fea_solve_dmg()

    t_end = time.perf_counter()
    
    total_time = t_end - t_start
    print("Time used:", total_time)
    
    with open(os.path.join(args['out_dir'], "statistics.txt"), "w") as f:
        f.write(f"Number of dofs: {n_total_dofs}\n")
        f.write(f"Total Solver time: {total_time} seconds\n")

    # Plot against the IGFEM reference, outside the timed region
    plot_comparison(args, vtk_mesh)
