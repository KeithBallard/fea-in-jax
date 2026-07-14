from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
# jax.config.update("jax_disable_jit", True)


def make_fabric(n_elements: int):
    x = np.linspace(0,1,n_elements+1)
    y = x*(1-x)/2.5

    points = np.vstack([x,y]).T
    fiber_offsets = np.concatenate([[0],np.cumsum([points.shape[0]])])

    fabric = VTMSFabric(
        name="test",
        material_ids=np.array([0]),
        diameters=np.array([0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, fiber_offsets.shape[0]-1]),
    )
    return fabric

def run_2D_flattening(
    n_elements: list[int],
    pseudoT:int,
    filename_base:str,
    contact_params: ContactParams,
    pre_strain:float,
    debug_info: DebugInfo | NullDebugInfo = NullDebugInfo
):
    """ """
    fabric = make_fabric(n_elements=n_elements)
    dyn_bcs = [[
        DirichletBC(index=0,          component=0, value=0, bc_type=BCType.NODE),
        DirichletBC(index=0,          component=1, value=0, bc_type=BCType.NODE),
        DirichletBC(index=n_elements, component=0, value=0, bc_type=BCType.NODE),
        DirichletBC(index=n_elements, component=1, value=0, bc_type=BCType.NODE),
    ]]

    E = 1e9
    A = 1
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[
            VTMSFiberMaterial(id=int(fabric.get_material_id(i)), E=E, A=A) for i in range(fabric.get_n_bundles())
        ],
        boundary_conditions=dyn_bcs,
        contact_options=contact_params,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            # linear_precond_type=PreconditionerType.JACOBI,
            nonlinear_max_iter=9,
            linear_max_iter=500,
            # max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
        ),
        pseudotime_iters=pseudoT,
        filename_base=filename_base,
        blow_up_threshold=1e6,
        plot_convergence=False,
        pre_strain=pre_strain,
        debug_info=debug_info,
    )
    if not isinstance(debug_info, NullDebugInfo): debug_info.file.close()
    return u,fabric,dyn_bcs

args = {
    'n_elements':10,
    'contact_params': ContactParams(
        self_adjacency_block    = 10000,
        contact_stiffness_model = __contact_stiffness_exponential,
        D_stiffness_to_E_ratio  = 0.25,
        contact_search_radius   = 0.2,
        M_to_D_ratio            = 1.25,
        M_stiffness_to_E_ratio  = 1.0/100.0
    ),
    'pseudoT':1,
    'filename_base':None,
    'debug_info':make_debug_info(
        flags = [
            (DebugOutputQuantities.NODE_SOLUTION,DebugOutputStage.NONLINEAR_SOLVE),
            (DebugOutputQuantities.GLOBAL_JACOBIAN_COO,DebugOutputStage.NONLINEAR_SOLVE),
        ],
        filename = 'prestrain/test_global_jac_coo.h5'
    )
}

def plot_nodes(fabric,file):
    plt.plot(*fabric.points.T,marker='x',label = 'nl_0')
    for k in list(file['ts_0'].keys())[1:]:
        plt.plot(*(fabric.points + file['ts_0'][k]['NODE_SOLUTION']['u'][:,:]).T,marker = 'x', label = k)
    plt.legend()
