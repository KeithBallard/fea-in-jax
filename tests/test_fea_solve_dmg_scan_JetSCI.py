import meshio
import numpy as np

from helper import *
from igfem_mesh import *
from typing import Any

import os
import pyvista as pv
import time
import jetsci

jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.post_pred import *
from fe_jax.write_vtk import *
from fe_jax.plot_results import plot_comparison
from fe_jax.linear_elasticity_dmg import *
from fe_jax.fea import solve_bvp_quasi_PETSc

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)
 

def test_fea_solve_dmg():
    args: dict[str, Any] = {}
    num_fib = '100'
    args['t_total']  = 500
    # args['dir_path'] = "4fib"
    args['dir_path'] = f"nonlinear_IGFEM_scan_t{args['t_total']}_{num_fib}fib_CG_JetSCI"
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # Create the directory (and any necessary parent directories)
    args['out_dir'] = os.path.join("tests/output",args['dir_path'])
    args['vtk_dir'] = os.path.join(args['out_dir'],"vtks")
    args['ref_dir'] = os.path.join("tests/IGFEM_ref",f"{num_fib}fib_t{args['t_total']}")
    os.makedirs(args['out_dir'], exist_ok=True)
    os.makedirs(args['vtk_dir'], exist_ok=True)

    # Read in the mesh (IGFEM)
    # mesh     = pv.read(f'tests/meshes/IGFEM_9fib_struct.vtk')
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

    dx_increment = args['strain_max'] * length / args['t_total']

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
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(233e3)      # E_xx
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(23.1e3)     # E_yy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(0.2)        # nu_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(8.96e3)     # G_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 4].set(8.27e3)     # G_yz
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
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,6]  .set(0)  # D
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

    print('Start Deformation Loop')
    
    u_history, ISV_be_history = solve_bvp_quasi_PETSc(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=dirichlet_bcs,
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
    n_total_dofs = u_history.shape[1]

    # Post-process: write VTK files
    print('Post-process')
    for i, bc in enumerate(u_history):
        time_step = i + 1
        print("Time step =", i+1)
        
        # Extract the state at time i
        u_i = u_history[i]
        ISV_be_i = jax.tree_util.tree_map(lambda x: x[i], ISV_be_history)

        # write and save to vtk
        vtk_mesh = write2VTK_ISV(args,mesh,u_i,ISV_be_i,batch_cell_ids)
        vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{time_step}.vtk")
    # zip_folder(args['vtk_dir'], args['vtk_dir']+'.zip')

    s11_global, s22_global, s12_global = compute_stress_strain_curve(ISV_be_history, element_batches, points)
    # Save param_hist and loss_hist
    np.save(os.path.join(args['out_dir'], "stress_strain_curve.npy"), np.array([dxs/length*100,s11_global]))
    np.save(os.path.join(args['out_dir'], "s_history.npy"), np.array([s11_global,s22_global,s12_global]))
    return n_total_dofs, args, vtk_mesh

if __name__ == "__main__":
    t_start = time.time()

    # with jax.profiler.trace("./jax-trace_new", create_perfetto_trace=True):
    n_total_dofs, args, vtk_mesh = test_fea_solve_dmg()

    t_end = time.time()
    total_time = t_end - t_start
    print("Time used:", total_time)
    
    with open(os.path.join(args['out_dir'], "statistics.txt"), "w") as f:
        f.write(f"Number of dofs: {n_total_dofs}\n")
        f.write(f"Total Solver time: {total_time} seconds\n")

    # Plot against the IGFEM reference, outside the timed region
    plot_comparison(args, vtk_mesh)
