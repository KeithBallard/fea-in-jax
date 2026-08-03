from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
# jax.config.update("jax_disable_jit", True)

def build_custom_hex(R,d,center=(0,0),rotation = 0):
    def horizontal_centers(n,d):
        if not n%2:
            hc = [(i+1/2)*d for i in range(n//2)]
            hc = [-i for i in reversed(hc)] + hc
        else:
            hc = [(i+1)*d for i in range(n//2)]
            hc = [-i for i in reversed(hc)] + [0] + hc
        return hc
    H = np.vstack(
        [
            np.vstack(
                [
                    [hc, d*i*(np.sqrt(3)/2)] for hc in horizontal_centers(r,d)
                ]
            )
            for i,r in enumerate(R)
        ]
    )
    # rotate
    R = np.array([[np.cos(rotation),-np.sin(rotation)],[np.sin(rotation),np.cos(rotation)]])
    H = np.matmul(R,H.T).T
    # recenter
    H = H - np.mean(H,axis=0) + np.array(center)
    return H

def make_single_fiber(
    n_elements: int,
    x0: tuple,
    xN: tuple,
    fiber_id: int,
    cell_shift: int,
    x_shift: np.ndarray | None = None,
    y_shift: np.ndarray | None = None,
    z_shift: np.ndarray | None = None,
):
    if x_shift is None:
        x_shift = np.zeros(n_elements + 1)
    if y_shift is None:
        y_shift = np.zeros(n_elements + 1)
    if z_shift is None:
        z_shift = np.zeros(n_elements + 1)
    points = np.vstack(
        (
            np.linspace(x0[0], xN[0], n_elements + 1) + x_shift,
            np.linspace(x0[1], xN[1], n_elements + 1) + y_shift,
            np.linspace(x0[2], xN[2], n_elements + 1) + z_shift,
        )
    ).T
    cells = np.array(
        [[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)],
        dtype=np.uint64,
    )
    fiber_ids = np.array([[fiber_id]] * len(points))
    cell_ids = np.array([[fiber_id]] * len(cells))
    return points, cells, fiber_ids, cell_ids


def make_bundle(
    n_elements: list[int],
    X0: list[tuple],
    XN: list[tuple],
    NeumannForce,
    x_shift: np.ndarray | None = None,
    y_shift: np.ndarray | None = None,
    z_shift: np.ndarray | None = None,
):
    point_blocks = []
    cell_blocks = []
    point_id_blocks = []
    cell_id_blocks = []

    bcs = []

    vertex_offset = 0

    for fiber_id, (n_el, x0, xN) in enumerate(zip(n_elements, X0, XN)):
        points_i, cells_i, point_ids_i, cell_ids_i = make_single_fiber(
            n_elements=n_el,
            x0=x0,
            xN=xN,
            fiber_id=fiber_id,
            cell_shift=vertex_offset,
            x_shift = x_shift,
            y_shift = y_shift,
            z_shift = z_shift,
        )

        point_blocks.append(points_i)
        cell_blocks.append(cells_i)
        point_id_blocks.append(point_ids_i)
        cell_id_blocks.append(cell_ids_i)
        bcs += [
            DirichletBC(bc_type=BCType.NODE, component=c, index=i, value=0.0)
            for c in (0, 1, 2)
            for i in (vertex_offset + 0, vertex_offset + n_el)
        ]
        vertex_offset += points_i.shape[0]

    points = np.vstack(point_blocks)
    cells = np.vstack(cell_blocks)
    point_ids = np.vstack(point_id_blocks).reshape(-1)
    cell_ids = np.vstack(cell_id_blocks).reshape(-1)

    fiber_offsets = np.concatenate(
        [
            [0],
            np.cumsum([b.shape[0] for b in point_blocks])
        ]
    )
    # fiber_offsets = np.cumsum([b.shape[0] for b in point_blocks])
    bundle = VTMSBundle(
        name="test",
        n_fibers=len(n_elements),
        material_id=np.array([0]),
        diameter=np.array([0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        # bundle_offsets=np.array([0, fiber_offsets.shape[0]]),
    )
    fabric = VTMSFabric(
        name="test",
        material_ids=np.array([0]),
        diameters=np.array([0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, fiber_offsets.shape[0]-1]),
    )
    return fabric,bcs

def run_threeFiberTow(
    n_elements: list[int],
    X0: list[tuple],
    XN: list[tuple],
    NeumannForce,
    contact_params,
    # solver_options: SolverOptions,
    filename_base = None,
    pre_strain: float | None = None,
    x_shift: np.ndarray | None = None,
    y_shift: np.ndarray | None = None,
    z_shift: np.ndarray | None = None,
    debug_info: DebugInfo | NullDebugInfo = NULL_DEBUG_INFO,
):
    """ """
    fabric, bcs = make_bundle(
        n_elements=n_elements,
        X0=X0,
        XN=XN,
        NeumannForce=NeumannForce,
        x_shift = x_shift,
        y_shift = y_shift,
        z_shift = z_shift,
    )
    d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_dist = d[d.nonzero()].min()
    solver_options=SolverOptions(
        # linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY ,
        # linear_solve_type=LinearSolverType.BICGSTAB_JAX_SCIPY ,
        # linear_precond_type=PreconditionerType.JACOBI,
        linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
        nonlinear_max_iter=100,
        linear_max_iter=200,
        max_linear_displacement=min(min_dist,fabric.diameters[0])/10,
        # max_linear_displacement=fabric.diameters[0]/10,
        max_backtracks=20,
    )
    if not isinstance(debug_info, NullDebugInfo):
        debug_info.file.attrs['contact_stiffness_model']        = contact_params.contact_constitutive_model.args[0].func.__name__.lstrip('_')
        debug_info.file.attrs['contact_D_stiffness_to_E_ratio'] = contact_params.D_stiffness_to_E_ratio
        debug_info.file.attrs['contact_search_radius']          = contact_params.contact_search_radius
        debug_info.file.attrs['contact_M_to_D_ratio']           = contact_params.M_to_D_ratio
        debug_info.file.attrs['contact_M_stiffness_to_E_ratio'] = contact_params.M_stiffness_to_E_ratio
        debug_info.file.attrs['contact_self_adjacency_block']   = contact_params.self_adjacency_block
        # debug_info.file.attrs['external_load_Fx_Fy']            = (0,-force)
        debug_info.file.attrs['solver_linear_solve_type']       = solver_options.linear_solve_type.name
        debug_info.file.attrs['solver_nonlinear_max_iter']      = solver_options.nonlinear_max_iter
        debug_info.file.attrs['solver_linear_max_iter']         = solver_options.linear_max_iter
        debug_info.file.attrs['solver_max_linear_displacement'] = solver_options.max_linear_displacement
        debug_info.file.attrs['points']                         = fabric.points
    dyn_bcs = []
    f_n = lambda z,nf : nf*(np.exp(-(4*z)**2) - np.exp(-16))/(1-np.exp(-16))
    nf_fiber = 2
    for nf in NeumannForce:
        temp_bcs =deepcopy(bcs)
        temp_bcs += [
            NeumannBC(
                bc_type   = BCType.NODE,
                component = 1,
                index     = fabric.fiber_offsets[nf_fiber] + i + 1,
                value     = -f_n(z,nf),
            )
            for i,z in enumerate(fabric.points[fabric.fiber_offsets[nf_fiber]+1:fabric.fiber_offsets[nf_fiber+1]-1,2])
        ]
        # temp_bcs += [
        #     NeumannBC(
        #         bc_type   = BCType.NODE,
        #         component = 1,
        #         index     = fabric.fiber_offsets[nf_fiber] + i + 1,
        #         # value     = -f_n(z,nf),
        #         value     = -f_n(fabric.points[fabric.fiber_offsets[nf_fiber]+int(n_elements[0]/2) +i,2],nf),
        #     )
        #     for i in range(-2,3)
        # ]
        dyn_bcs.append(temp_bcs)


    E = 1e9
    A = (fabric.diameters[0]/2)**2*np.pi
    print(f"EA/N = {E*A/NeumannForce}")
    print(f"{min(min_dist/2,fabric.diameters[0]/2)}")
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[VTMSFiberMaterial(id=0, E=E, A=A)],
        boundary_conditions=dyn_bcs,
        solver_options=solver_options,
        contact_options=contact_params,
        plot_convergence=False,
        filename_base=filename_base,
        pseudotime_iters=len(dyn_bcs),
        pre_strain=pre_strain,
        debug_info=debug_info,
        blow_up_threshold=10,
    )
    u = u.reshape((-1,3))

    D_D = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_d = D_D[D_D.nonzero()].min()
    if not isinstance(debug_info, NullDebugInfo):
        print('close debug HDF5 file')
        debug_info.file.close()
    return u,fabric,dyn_bcs

# u,f = run_threeFiberTow(
#     n_elements=[10, 10, 10],
#     X0=[[0, 0, -1], [0.1, 0, -1], [0.5 * 0.1, np.sqrt(3) / 2 * 0.1, -1]],
#     XN=[[0, 0, 1], [0.1, 0, 1], [0.5 * 0.1, np.sqrt(3) / 2 * 0.1, 1]],
#     contact_search_radius=0.25,
#     NeumannForce = 1E5
# )
VTMS_args = {
    'n_elements':[40]*3,
    'X0':[[i[0],i[1],-1] for i in build_custom_hex([2,1],0.1)],
    'XN':[[i[0],i[1],1] for i in build_custom_hex([2,1],0.1)],
    'NeumannForce':[(i+1)*1e3 for i in range(10)],
    # 'NeumannForce':np.concatenate(
    #     (
    #         [(i+1)*1000 for i in range(3)],
    #         [3000+(i+1)*250 for i in range(11)],
    #         [(i+1)*1000 for i in range(5,10)]
    #     )
    # ),
    # 'filename_base': 'ThreeFiberSpread/Jul30/quadratic',
    'filename_base': None,
    'contact_params': ContactParams(
        self_adjacency_block       = 10000,
        contact_constitutive_model = elastic_contact_truss_piecewise_quadratic,
        D_stiffness_to_E_ratio     = 1.,
        M_to_D_ratio               = 1.05,
        C_to_D_ratio               = 1.0,
        M_stiffness_to_E_ratio     = 0.000,
        contact_search_alpha       = 1.4,
    ),
}
# run_threeFiberTow(**VTMS_args)
# debug_info=make_debug_info(
#     flags = [
#         (DebugOutputQuantities.NODE_SOLUTION,DebugOutputStage.NONLINEAR_SOLVE),
#         (DebugOutputQuantities.NODE_RESIDUAL,DebugOutputStage.NONLINEAR_SOLVE),
#         (DebugOutputQuantities.ELEMENT_RESIDUAL,DebugOutputStage.NONLINEAR_SOLVE),
#     ],
#     filename = 'contact/twoD_triangle.h5'
# )
def get_min(fabric,i,j):
    fi = fabric.get_fiber_points(0,i)
    fj = fabric.get_fiber_points(0,j)
    D = np.linalg.norm(fi[None,:,:] - fj[:,None,:],axis=-1)
    return D[D.nonzero()].min()

def get_mins(fabric):
    n = fe.get_n_fibers_in_bundle(0)
    M = []
    for i in range(n):
        for j in range(i+1,n):
            print(f"({i},{j}) - {get_min(fabric,i,j)}")
            M.append(get_min(fabric,i,j))
    return np.array(M).min()

# get_mins(fl)
# get_mins(fp)
# get_mins(fe)

def plot_horizontal_displacement(filename,max_range,min_range=0):
    D = []
    plt.figure(figsize=[12,8])
    plt.subplot(121)
    for i in range(0,max_range):
        mesh = meshio.read(f"output/contact/ThreeFiberSpread/{filename}_wireframe_{i}.vtk")
        center_index = np.abs(mesh.points[:,2])<0.01
        P = mesh.points[center_index]
        D.append(P)
        p = P[:,[0,1]]
        # p[:,1] -= P[2,1]
        if i>=min_range: plt.scatter(*(p).T,label = f"t_i = {i}")
    plt.grid()
    plt.legend(loc = 'center left',bbox_to_anchor=(1.02,0.5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')

    plt.subplot(122)
    H_tic = np.array(D)
    plt.plot(range(H_tic.shape[0]),H_tic[:,0,0]-H_tic[0,0,0],color='blue', label = 'node 0', marker = 'x')
    plt.plot(range(H_tic.shape[0]),H_tic[:,1,0]-H_tic[0,1,0],color='gray', label = 'node 1', marker = 'x')
    plt.plot(range(H_tic.shape[0]),H_tic[:,2,0]-H_tic[0,2,0],color='red',label = 'node 2', marker = 'x')
    plt.xlabel('pseudo-time index')
    plt.ylabel('displacement in the x direciton')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_output(f"contact/{filename}_horizontal_displacement.pdf"))
    plt.close()
