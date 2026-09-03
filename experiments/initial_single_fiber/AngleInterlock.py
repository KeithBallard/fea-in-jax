from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import scipy as sp
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
    n_elements: int, x0: tuple, xN: tuple, fiber_id: int, cell_shift: int
):
    points = np.vstack(
        (
            np.linspace(x0[0], xN[0], n_elements + 1),
            np.linspace(x0[1], xN[1], n_elements + 1),
            np.linspace(x0[2], xN[2], n_elements + 1),
        )
    ).T
    cells = np.array(
        [[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)],
        dtype=np.uint64,
    )
    fiber_ids = np.array([[fiber_id]] * len(points))
    cell_ids = np.array([[fiber_id]] * len(cells))
    return points, cells, fiber_ids, cell_ids


def make_bundle(n_elements: list[int], X0: list[tuple], XN: list[tuple],diameter: float):
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
    fabric = VTMSFabric(
        name="test",
        material_ids=np.array([0]),
        diameters=np.array([diameter]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, fiber_offsets.shape[0]-1]),
    )
    return fabric,bcs

def make_cyl_mold(xy_center,R,L,dx):
    d_theta = 2*np.arcsin(dx/(2*R))
    theta = np.linspace(0,2*np.pi,int(2*np.pi/d_theta))
    X = xy_center[0] + R*np.cos(theta[:-1])
    Y = xy_center[1] + R*np.sin(theta[:-1])
    Z = np.linspace(-L/2,L/2,int(L/dx))
    P = np.vstack([np.vstack([X,Y,np.full((Y.shape[0],),z)]).T for z in Z])

    kd_tree = sp.spatial.cKDTree(P)
    C = np.array(list(kd_tree.query_pairs(r=1.1*dx)))
    if C.shape[0] == 0:
        C = np.zeros((0,2), dtype = np.int32)
    # d = P[:,None,:] - P[None,:,:]
    # dist = jnp.linalg.norm(d,axis=-1)
    # dist_mask = dist <= 1.1*dx
    # upper_mask = jnp.triu(jnp.ones((P.shape[0],P.shape[0]), dtype=bool),k=1)
    # mask = dist_mask & upper_mask
    # C = np.vstack(mask.nonzero()).T
    return P,C

def make_boundary_conditions(
    fabric,
):
    bcs = []

    # Fix both ends of original bundle fibers.
    for i in fabric.fiber_offsets[:-1]:
        for c in range(3):
            bcs.append(DirichletBC(index=i, component=c, value=0.0, bc_type=BCType.NODE))

    for i in fabric.fiber_offsets[1:]:
        for c in range(3):
            bcs.append(DirichletBC(index=i - 1, component=c, value=0.0, bc_type=BCType.NODE))

    return bcs

def make_bc_schedule(
    fabric,
    old_fibers_n: int,
    old_points_n: int,
    fabric_n: int,
    n_load_steps: int,
    dir_step: float,
    schedule: list[str] | tuple[str, ...] = ("LOAD", "RELAX"),
):
    dyn_bcs = []
    labels = []

    for k in range(1, n_load_steps + 1):
        pin_y = k * dir_step

        for stage in schedule:
            stage = stage.upper()
            if stage not in ("LOAD", "RELAX"):
                raise ValueError(f"Unknown schedule stage: {stage}")

            dyn_bcs.append(make_boundary_conditions(
                fabric=fabric,
                old_fibers_n=old_fibers_n,
                old_points_n=old_points_n,
                fabric_n=fabric_n,
                pin_y=pin_y,
            ))
            labels.append((stage, k))

    return dyn_bcs, labels

def refine_fiber(
    points: jnp.ndarray,
    n_elements: int | None,
    dX: float | None= None,
):
    d = np.linalg.norm(np.diff(points,axis=0),axis=1)
    L = np.concatenate([[0.0], np.cumsum(d)])
    if n_elements is None and dX is None:
        raise(RuntimeError("Must provide number of elements in fiber (n_elements) or length of elements (dX), but you provided neither."))
    if n_elements is not None and dX is not None:
        raise(RuntimeError("Must provide either the number of elements in fiber (n_elements) or length of elements (dX), but not both."))
    if n_elements is None:
        n_elements = int(L[-1]/dX + 1)
    spline = sp.interpolate.CubicSpline(L,points)
    y = spline(np.linspace(0,L[-1],n_elements+1))
    return y



def refine_tow(fabric_in,pattern):
    def circle_pack(pattern,outer_diameter):
        centers = build_custom_hex(pattern,1.0)
        temp_diam = 2*(np.linalg.norm(centers,axis=1).max()+0.5)
        diam = outer_diameter/temp_diam
        centers = build_custom_hex(pattern,diam)
        return centers,diam
    refined_N = sum(pattern)

    point_list = []
    diameters = []

    e1 = np.array([1,0,0])
    for i in range(fabric_in.get_n_bundles()):
        p = fabric_in.get_fiber_points(i,0)
        grad = np.gradient(p, axis = 0, edge_order=2)
        grad /= np.linalg.norm(grad, axis =1,keepdims=True)
        rots = [sp.spatial.transform.Rotation.align_vectors([g],[e1])[0] for g in grad]
        mats = np.stack([r.as_matrix() for r in rots])

        c,d =  circle_pack(pattern,fabric_in.get_diameter(i))
        for center in c:
            rotated_center = mats @ np.array([0,center[0],center[1]])
            point_list.append(p + rotated_center)
        diameters.append(d)

    fiber_offsets = np.concatenate([[0],np.cumsum([point_set.shape[0] for point_set in point_list])])
    points = np.vstack(point_list)
    diameters = np.array(diameters)
    fabric = VTMSFabric(
        name="RefinedTowsInFabric",
        material_ids=np.array([fabric_in.get_material_id(i) for i in range(fabric_in.get_n_bundles())]),
        diameters=diameters,
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=refined_N*fabric_in.bundle_offsets,
    )
    return fabric

def refine_fabric(fabric_in, dX, tow_n = 1, n_elements = None):
    point_list = [refine_fiber(fabric_in.get_fiber_points(i,0),n_elements=n_elements,dX=dX) for i in range(fabric_in.get_n_bundles())]
    fiber_offsets = np.concatenate([[0],np.cumsum([point_set.shape[0] for point_set in point_list])])
    points = np.vstack(point_list)

    fabric = VTMSFabric(
        name="RefinedFabric",
        material_ids=np.array([fabric_in.get_material_id(i) for i in range(fabric_in.get_n_bundles())]),
        diameters=np.array([fabric_in.get_diameter(i) for i in range(fabric_in.get_n_bundles())]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=fabric_in.bundle_offsets,
    )
    return fabric

def run_tension(
    fabric: VTMSBundle | VTMSFabric,
    contact_params: ContactParams,
    pseudoT:int,
    filename_base =None,
    pre_strain: float | None = None,
    debug_info: DebugInfo | NullDebugInfo = NULL_DEBUG_INFO,
    rigid_mold = None,
):
    """ """

    fabric = refine_fabric(fabric,dX=0.25)
    fabric = refine_tow(fabric, [2,3,2])
    if not isinstance(debug_info, NullDebugInfo):
        debug_info.file.attrs['contact_stiffness_model']        = contact_params.contact_constitutive_model.args[0].func.__name__.lstrip('_')
        debug_info.file.attrs['contact_D_stiffness_to_E_ratio'] = contact_params.D_stiffness_to_E_ratio
        debug_info.file.attrs['contact_M_to_D_ratio']           = contact_params.M_to_D_ratio
        debug_info.file.attrs['contact_M_stiffness_to_E_ratio'] = contact_params.M_stiffness_to_E_ratio
        debug_info.file.attrs['contact_self_adjacency_block']   = contact_params.self_adjacency_block


    # dyn_bcs, stage_labels = make_bc_schedule(
    #     fabric=fabric,
    #     old_fibers_n=old_fibers_n,
    #     old_points_n=old_points_n,
    #     fabric_n=fabric_n,
    #     n_load_steps=pseudoT,
    #     dir_step=dir_step,
    #     schedule=("LOAD"),
    # )
    dyn_bcs = [make_boundary_conditions(fabric)]*pseudoT

    # d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    # min_dist = d[d.nonzero()].min()


    # dyn_bcs = []
    # for ii in range(pseudoT):
    #     temp_bcs =deepcopy(bcs)
    #     for temp_bc,control_bc in zip(temp_bcs,bcs):
    #         temp_bc.value = control_bc.value*(ii+1)
    #     dyn_bcs.append(temp_bcs)
    # return fabric,rigid_mold,dyn_bcs
    print('dynamic boundary conditions generated!')

    E = 0.5
    tow_A = (fabric.get_diameter(0)/2)**2*np.pi
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        rigid_mold=rigid_mold,
        materials=[
            VTMSFiberMaterial(
                id=fabric.get_material_id(i),
                E=E,
                A=np.pi*(fabric.get_diameter(i)/2)**2,
            ) for i in range(fabric.get_n_bundles())
        ],
        boundary_conditions=dyn_bcs,
        contact_options=contact_params,
        solver_options=SolverOptions(
            # linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.BICGSTAB_JAX_SCIPY,
            linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=75,
            linear_max_iter=200,
            damp_Newton_diag=1.0,
            # nonlinear_relative_tol=.0001,
            # max_linear_displacement=0.5,
            max_backtracks=20,
            # linear_absolute_tol=3.16e-3,
            # max_linear_displacement=min(min_dist,fabric.diameters[0])/2,
        ),
        plot_convergence=False,
        filename_base=filename_base,
        pseudotime_iters=len(dyn_bcs),
        blow_up_threshold=10,
        pre_strain=pre_strain,
        debug_info=debug_info,
    )
    u = u.reshape((-1,3))
    # fabric.points = fabric.points + u[:fabric_n,:]

    if not isinstance(debug_info, NullDebugInfo):
        print('close debug HDF5 file')
        debug_info.file.close()
    return u,fabric,dyn_bcs

args = {
    'fabric':read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab"),
    'filename_base': 'FabricExample/Aug4/tensioning_higherPreStrain_damped_NL250_tanh_refined_tow_planar_totalLagrangian',
    # 'filename_base': None,
    'pseudoT': 1,
    'pre_strain':-0.141373887*20,
    'contact_params': ContactParams(
        self_adjacency_block    = 10000,
        contact_constitutive_model = elastic_contact_truss_tanh,
        D_stiffness_to_E_ratio  = 6.0,
        # M_stiffness_to_E_ratio  = 1e-6,
        M_stiffness_to_E_ratio  = 0.001,
        M_to_D_ratio            = 1.00,
        C_to_D_ratio            = 0.8,
        contact_search_alpha    = 2.0,
    ),
}

debug_info=make_debug_info(
    flags = [
        (DebugOutputQuantities.GLOBAL_JACOBIAN_COO,DebugOutputStage.NONLINEAR_SOLVE),
        (DebugOutputQuantities.NODE_SOLUTION,DebugOutputStage.NONLINEAR_SOLVE),
        (DebugOutputQuantities.NODE_RESIDUAL,DebugOutputStage.NONLINEAR_SOLVE),
    ],
    filename = args['filename_base'] + '.h5'
)
