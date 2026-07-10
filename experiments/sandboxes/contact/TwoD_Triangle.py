from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ani
# jax.config.update("jax_disable_jit", True)

def run_2D_triangle(
    pseudoT:int,
    filename_base:str,
    contact_params: ContactParams,
    force: float,
    fiber_offsets: List = [0,1,5],
    debug_info: DebugInfo | NullDebugInfo = NullDebugInfo
):
    """ """
    fiber_offsets = np.array(fiber_offsets)
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
        fiber_offsets = fiber_offsets,
        bundle_offsets = np.array([0,fiber_offsets.shape[0]-1])
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

    solver_options=SolverOptions(
        # linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
        # linear_precond_type=PreconditionerType.JACOBI,
        nonlinear_max_iter=50,
        linear_max_iter=500,
        # max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
        # max_linear_displacement=0.02,
    )

    if not isinstance(debug_info, NullDebugInfo):
        debug_info.file.attrs['contact_stiffness_model']        = contact_params.contact_stiffness_model.__name__
        debug_info.file.attrs['contact_D_stiffness_to_E_ratio'] = contact_params.D_stiffness_to_E_ratio
        debug_info.file.attrs['contact_search_radius']          = contact_params.contact_search_radius
        debug_info.file.attrs['contact_M_to_D_ratio']           = contact_params.M_to_D_ratio
        debug_info.file.attrs['contact_M_stiffness_to_E_ratio'] = contact_params.M_stiffness_to_E_ratio
        debug_info.file.attrs['contact_self_adjacency_block']   = contact_params.self_adjacency_block
        debug_info.file.attrs['external_load_Fx_Fy']            = (0,-force)
        debug_info.file.attrs['solver_linear_solve_type']       = solver_options.linear_solve_type.name
        debug_info.file.attrs['solver_nonlinear_max_iter']      = solver_options.nonlinear_max_iter
        debug_info.file.attrs['solver_linear_max_iter']         = solver_options.linear_max_iter
        debug_info.file.attrs['solver_max_linear_displacement'] = solver_options.max_linear_displacement
        debug_info.file.attrs['points']                         = fabric.points
    E = 1e9
    A = np.pi*(0.1/2)**2
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[
            VTMSFiberMaterial(id=int(fabric.get_material_id(i)), E=E, A=A) for i in range(fabric.get_n_bundles())
        ],
        boundary_conditions= dyn_bcs,
        contact_options    = contact_params,
        solver_options     = solver_options,
        pseudotime_iters   = pseudoT,
        filename_base      = filename_base,
        blow_up_threshold  = 1e6,
        plot_convergence   = False,
        debug_info         = debug_info,
    )
    if not isinstance(debug_info, NullDebugInfo):
        print('close debug HDF5 file')
        debug_info.file.close()
    return u.reshape(-1,fabric.points.shape[1]),fabric,dyn_bcs

args = {
    'contact_params': ContactParams(
        self_adjacency_block    = 2,
        contact_stiffness_model = contact_stiffness_exponential,
        D_stiffness_to_E_ratio  = 100,
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

def plot_nodes(filename, framerate = 5, nl_max = 10000):
    from matplotlib.patches import Circle

    f_temp = h5py.File(f'debug/contact/{filename}.h5')
    C = f_temp['ts_0/nl_0/ELEMENT_RESIDUAL/batch_1_connectivity'][:]

    contact_model_name = f_temp.attrs['contact_stiffness_model']
    if isinstance(contact_model_name, bytes):
        contact_model_name = contact_model_name.decode()
    contact_stiffness_model = globals()[contact_model_name]

    contact_search_radius = float(f_temp.attrs['contact_search_radius'])
    contact_D_stiffness_to_E_ratio = float(f_temp.attrs['contact_D_stiffness_to_E_ratio'])
    contact_M_to_D_ratio = float(f_temp.attrs['contact_M_to_D_ratio'])
    contact_M_stiffness_to_E_ratio = float(f_temp.attrs['contact_M_stiffness_to_E_ratio'])

    if contact_model_name == 'contact_stiffness_linear':
        contact_params_plot = jnp.array([1.0, 1.0, contact_search_radius])
    elif contact_model_name == 'contact_stiffness_constant':
        contact_params_plot = jnp.array([1.0])
    elif contact_model_name in ('contact_stiffness_piecewise_linear', 'contact_stiffness_exponential'):
        contact_params_plot = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                contact_M_to_D_ratio,
                contact_M_stiffness_to_E_ratio / contact_D_stiffness_to_E_ratio,
            ]
        )
    else:
        raise ValueError(f"Unknown contact stiffness model: {contact_model_name}")

    def eval_contact_stiffness(d):
        k = np.asarray(contact_stiffness_model(jnp.asarray(d), contact_params_plot))
        if np.ndim(k) == 0:
            k = np.full_like(np.asarray(d, dtype=float), float(k), dtype=float)
        return k

    d_grid = np.linspace(0.0, contact_search_radius, 400)
    k_grid = eval_contact_stiffness(d_grid)

    NL = list(f_temp['ts_0'])
    nl_max = min(nl_max,np.array([int(NL[k][3:]) for k in range(len(NL))]).max())
    u = np.array([f_temp[f'ts_0/nl_{k}/NODE_SOLUTION/u'][:] for k in range(1,nl_max)])
    r = np.array([f_temp[f'ts_0/nl_{k}/NODE_RESIDUAL/residual'][:] for k in range(1,nl_max)])


    u = np.concatenate((np.zeros((1,u.shape[1],u.shape[2])), u))

    p = f_temp.attrs['points'] + u

    el_F = []
    el_d = []
    for i in range(nl_max):
        F = f_temp[f'ts_0/nl_{i}/ELEMENT_RESIDUAL/batch_1_residual_wo_constraints'][:]
        X = p[i][C]
        D = X[:,-1,:] - X[:,0,:]
        n = D / np.linalg.norm(D, axis = 1, keepdims=True)
        el_d.append(np.linalg.norm(D, axis = 1))
        el_F.append(0.5*np.sum((F[:,1,:] - F[:,0,:]) * n, axis = 1))
    el_F = np.array(el_F)
    el_d = np.array(el_d)

    node_idx = [0,1,4]
    x_min,x_max = p[:,node_idx,0].min(), p[:,node_idx,0].max()
    y_min,y_max = p[:,node_idx,1].min(), p[:,node_idx,1].max()

    fig = plt.figure(figsize=[10,5],constrained_layout = True)
    gs = fig.add_gridspec(3,2)

    ax_main = fig.add_subplot(gs[:,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[2,1])

    ax1.plot(p[:,0,1],marker='x', label = 'X_0,y', color = 'tab:orange')
    ax1.plot(p[:,4,0] - p[:,1,0],marker='x',label = '|X_2-X_1|', color = 'tab:blue')
    ax1.set_ylabel('distance')
    ax1.set_xlabel('nonlinear iteration')
    ax1.legend()
    current_iter_line_ax1 = ax1.axvline(0, color='k', linestyle='--', linewidth=1)

    ax2.plot(el_F[:,0],marker='x', label = '0-1, 0-2', color = 'tab:purple')
    ax2.plot(el_F[:,2],marker='x', label = '1-2', color = 'tab:blue')
    ax2.set_ylabel('Axial force')
    ax2.set_xlabel('nonlinear iteration')
    ax2.legend()
    current_iter_line_ax2 = ax2.axvline(0, color='k', linestyle='--', linewidth=1)

    if contact_model_name == 'contact_stiffness_exponential':
        ax3.semilogy(d_grid, k_grid, color='0.25', lw=2, )
    else:
        ax3.plot(d_grid, k_grid, color='0.25', lw=2, )
    # ax3.set_ylabel(contact_model_name)
    contact_model_label = " ".join(contact_model_name.split("_")[:2])
    contact_model_tail = " ".join(contact_model_name.split("_")[2:])
    if contact_model_tail:
        contact_model_label += "\n" + contact_model_tail
    ax3.set_ylabel(contact_model_label)
    ax3.set_xlabel('distance')
    ax3.set_xlim(0.0, contact_search_radius)
    ax3.set_ylim(0.0, max(1.0, float(np.max(k_grid)) * 1.05))
    contact_markers_purple = ax3.scatter([], [], s=50, color='tab:purple', zorder=3, label='0-1 / 0-2')
    contact_markers_blue = ax3.scatter([], [], s=50, color='tab:blue', zorder=3, label='1-2')
    ax3.legend()


    circles = []
    labels = []
    for n, circle_center in enumerate(f_temp.attrs['points'][node_idx,:]):
        circ = Circle(circle_center, radius = 0.5, fill = False)
        ax_main.add_patch(circ)
        circles.append(circ)

        txt = ax_main.text(circle_center[0], circle_center[1], str(n), ha='center', va = 'center')
        labels.append(txt)

    ax_main.set_title('nonlinear iteration: 0')
    ax_main.set_xlim(x_min-0.5, x_max+0.5)
    ax_main.set_ylim(y_min-0.5, y_max+0.5)
    ax_main.set_aspect('equal')


    def func(i):
        for circ,txt,circle_center in zip(circles,labels, p[i,node_idx,:]):
            circ.center = circle_center
            txt.set_position(circle_center)
        ax_main.set_title(f'nonlinear iteration: {i}')
        current_iter_line_ax1.set_xdata([i, i])
        current_iter_line_ax2.set_xdata([i, i])
        k_i = eval_contact_stiffness(el_d[i])
        contact_markers_purple.set_offsets(np.column_stack([el_d[i, :2], k_i[:2]]))
        contact_markers_blue.set_offsets(np.column_stack([el_d[i, 2:3], k_i[2:3]]))
        return circles + labels + [
            current_iter_line_ax1,
            current_iter_line_ax2,
            contact_markers_purple,
            contact_markers_blue,
        ]

    anim = ani.FuncAnimation(fig, func, frames = range(u.shape[0]), blit=True)
    Writ = ani.FFMpegWriter(fps=framerate, metadata=dict(artist='nathan'))
    anim.save(get_debug_output(f'contact/{filename}.mp4'), writer = Writ)
    plt.close()
