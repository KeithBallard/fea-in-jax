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

def run_mold(
    fabric: VTMSBundle | VTMSFabric,
    contact_params: ContactParams,
    pseudoT:int,
    dir_step:float,
    filename_base =None,
    rigid_mold_params: list | None = None,
    pre_strain: float | None = None,
):
    """ """
    # fabric, bcs = make_bundle(n_elements=n_elements, X0=X0, XN=XN,diameter=diameter)
    fabric_n = fabric.points.shape[0]
    rigid_mold_id = np.sum([fabric.get_n_fibers_in_bundle(i) for i in range(fabric.get_n_bundles())])+1

    bcs = [DirichletBC(index = i, component = c, value = 0, bc_type=BCType.NODE) for i in fabric.fiber_offsets[:-1] for c in range(3)]
    bcs += [DirichletBC(index = i-1, component = c, value = 0, bc_type=BCType.NODE) for i in fabric.fiber_offsets[1:] for c in range(3)]


    d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_dist = d[d.nonzero()].min()
    if rigid_mold_params:
        mold_points,mold_connections = make_cyl_mold(*rigid_mold_params)
        rigid_mold = RigidMold(
            points = mold_points,
            point_ids= np.full((mold_points.shape[0],),rigid_mold_id),
            connections = mold_connections,
        )
        bcs += [DirichletBC(index = i, component = 0, value = 0, bc_type=BCType.NODE) for i in range(fabric_n,fabric_n + rigid_mold.points.shape[0])]
        bcs += [DirichletBC(index = i, component = 1, value = dir_step, bc_type=BCType.NODE) for i in range(fabric_n,fabric_n + rigid_mold.points.shape[0])]
        bcs += [DirichletBC(index = i, component = 2, value = 0, bc_type=BCType.NODE) for i in range(fabric_n,fabric_n + rigid_mold.points.shape[0])]
    else:
        rigid_mold = None


    dyn_bcs = []
    for ii in range(pseudoT):
        temp_bcs =deepcopy(bcs)
        for temp_bc,control_bc in zip(temp_bcs,bcs):
            temp_bc.value = control_bc.value*(ii+1)
        dyn_bcs.append(temp_bcs)
    # return fabric,rigid_mold,dyn_bcs
    print('dynamic boundary conditions generated!')

    E = 0.5
    A = (fabric.diameter/2)**2*np.pi
    # print(f"{min(min_dist/2,fabric.diameters[0]/2)}")
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        rigid_mold=rigid_mold,
        materials=[VTMSFiberMaterial(id=0, E=E, A=A)],
        boundary_conditions=dyn_bcs,
        contact_options=contact_params,
        solver_options=SolverOptions(
            # linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=600,
            linear_max_iter=500,
            # max_linear_displacement=min(min_dist,fabric.diameters[0])/2,
        ),
        plot_convergence=False,
        filename_base=filename_base,
        pseudotime_iters=len(dyn_bcs),
        blow_up_threshold=1e3,
        pre_strain=pre_strain,
    )
    u = u.reshape((-1,3))
    # fabric.points = fabric.points + u[:fabric_n,:]

    return u,fabric,rigid_mold

args = {
    'fabric': read_fib('experiments/sandboxes/rigidmold/pin_and_bundle.bdb'),
    'filename_base': 'rigid_mold/EZ/19Tow_D0p02_M0p001_MDratio1p05_piecewise_',
    'pseudoT': 30,
    'rigid_mold_params': ((5.,0.86),0.4,4.0,0.2),
    'dir_step':-0.02,
    'pre_strain':-0.141373887,
    'contact_params': ContactParams(
        self_adjacency_block    = 10000,
        contact_constitutive_model = elastic_contact_truss_constant,
        D_stiffness_to_E_ratio  = 1.0,
        contact_search_radius   = 0.3,
        M_to_D_ratio            = 1.025,
        M_stiffness_to_E_ratio  = 0.001
    ),
}
