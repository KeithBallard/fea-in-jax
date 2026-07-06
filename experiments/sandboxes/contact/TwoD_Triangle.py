from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
# jax.config.update("jax_disable_jit", True)



def run_2D_triangle(
    pseudoT:int,
    filename_base:str,
    contact_params: ContactParams,
    force: float,
    debug_info: DebugInfo | NullDebugInfo = NullDebugInfo
):
    """ """
    fabric = VTMSFabric(
        name="TwoD_Triangle",
        material_ids=np.array([0]),
        diameters=np.array([1.0]),
        points = np.array([
            [0,np.sqrt(3)/2],
            [-0.5,0.],
            [-0.5,-2.],
            [0.5,-2.],
            [0.5,0.],
        ]),
        fiber_offsets = np.array([0,1,2,4,5]),
        bundle_offsets = np.array([0,4])
    )

    dyn_bcs = [[
        DirichletBC(index=0, component=0, value=0, bc_type=BCType.NODE),
        NeumannBC(index=0, component=1, value=-force, bc_type=BCType.NODE),
        DirichletBC(index=1, component=1, value=0, bc_type=BCType.NODE),
        DirichletBC(index=4, component=1, value=0, bc_type=BCType.NODE),
        DirichletBC(index=2, component=0, value=0, bc_type=BCType.NODE),
        DirichletBC(index=2, component=1, value=0, bc_type=BCType.NODE),
        DirichletBC(index=3, component=0, value=0, bc_type=BCType.NODE),
        DirichletBC(index=3, component=1, value=0, bc_type=BCType.NODE),
    ]]

    E = 1e9
    A = np.pi*(0.1/2)**2
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[
            VTMSFiberMaterial(id=int(fabric.get_material_id(i)), E=E, A=A) for i in range(fabric.get_n_bundles())
        ],
        boundary_conditions=dyn_bcs,
        contact_options=contact_params,
        solver_options=SolverOptions(
            # linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            # linear_precond_type=PreconditionerType.JACOBI,
            nonlinear_max_iter=200,
            linear_max_iter=500,
            # max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
            max_linear_displacement=0.2,
        ),
        pseudotime_iters=pseudoT,
        filename_base=filename_base,
        blow_up_threshold=1e6,
        plot_convergence=False,
        debug_info=debug_info,
    )
    if not isinstance(debug_info, NullDebugInfo):
        print('close debug HDF5 file')
        debug_info.file.close()
    return u.reshape(-1,fabric.points.shape[1]),fabric,dyn_bcs

args = {
    'contact_params': ContactParams(
        self_adjacency_block    = 2,
        contact_stiffness_model = contact_stiffness_exponential,
        D_stiffness_to_E_ratio  = 0.25,
        contact_search_radius   = 1.5,
        M_to_D_ratio            = 1.1,
        M_stiffness_to_E_ratio  = 1.0/100.0
    ),
    'pseudoT':1,
    'filename_base':None,
    # 'debug_info':NULL_DEBUG_INFO,
    'debug_info':make_debug_info(
        flags = [
            (DebugOutputQuantities.NODE_SOLUTION,DebugOutputStage.NONLINEAR_SOLVE),
            (DebugOutputQuantities.NODE_RESIDUAL,DebugOutputStage.NONLINEAR_SOLVE),
            (DebugOutputQuantities.ELEMENT_RESIDUAL,DebugOutputStage.NONLINEAR_SOLVE),
        ],
        filename = 'contact/twoD_triangle.h5'
    )
}
