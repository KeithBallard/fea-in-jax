import meshio
import numpy as np

# import fea_traditional as test
from helper import *
from igfem_mesh import *

import os
import pyvista as pv
import time
# jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.post_pred import *
from fe_jax.write_vtk import *
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)
 

def test_fea_solve_dmg():
    args = {}
    num_fib = 100    # 1, 2, 4, 9, 16, 23, 33, 36, 46, 49, 60, 64, 100
    args['t_total']  = 500
    # args['dir_path'] = "debug"
    # args['dir_path'] = f"CG_t{args['t_total']}_{num_fib}fib"
    args['dir_path'] = f"SP_t{args['t_total']}_{num_fib}fib"
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # Create the directory (and any necessary parent directories)
    args['out_dir'] = os.path.join("tests/output",args['dir_path'])
    args['vtk_dir'] = os.path.join(args['out_dir'],"vtks")
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

    print_cell_ID = 12
    print_cell,fib_matrix_shape = find_print_cell_idx(print_cell_ID, batch_cell_ids)

    u_prev = jnp.array(jnp.reshape(vtk_mesh['displacement'][:,:2],-1), dtype=jnp.float64)
    # u_prev = None

    print('Start Deformation Loop')
    for i in range(1,args['t_total']+1):
        dirichlet_values[n_LHS:n_vals] = strain_increment * i
        for bc in range(len(dirichlet_bcs)):
            dirichlet_bcs[bc].value = dirichlet_values[bc]

        # Solve the boundary value problem
        u, residual, element_batches = solve_bvp(
            element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=element_batches,
            u_0_g=u_prev,
            boundary_conditions=dirichlet_bcs,
            solver_options=SolverOptions(
                linear_precond_type=PreconditionerType.JACOBI,
                # linear_solve_type=LinearSolverType.CG_JAX_SCIPY,#_W_INFO,
                linear_solve_type=LinearSolverType.SPSOLVE_CUPY ,
                linear_max_iter=10000,

            ),
        )

        u_prev = u
        print("Time step =", i)
        # Update displacements to be 3D for VTK on GPU, then transfer once to host
        # u_full = np.array(jnp.zeros((points.shape[0], 3)).at[:, :U].set(u.reshape(-1, U)))

        # # write and save to vtk
        vtk_mesh = write2VTK_avg(args,mesh,u,element_batches,batch_cell_ids)
        vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{i}.vtk")
    # zip_folder(args['vtk_dir'], args['vtk_dir']+'.zip')

    n_total_dofs = u.shape[0]

    return n_total_dofs, args['out_dir']


if __name__ == "__main__":
    t_start = time.perf_counter()

    # with jax.profiler.trace("./jax-trace", create_perfetto_trace=True):
    n_total_dofs, out_dir = test_fea_solve_dmg()

    t_end = time.perf_counter()
    total_time = t_end - t_start
    print("Time used:", total_time)

    with open(os.path.join(out_dir, "statistics.txt"), "w") as f:
        f.write(f"Number of dofs: {n_total_dofs}\n")
        f.write(f"Total Solver time: {total_time} seconds\n")