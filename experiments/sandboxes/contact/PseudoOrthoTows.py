from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
# jax.config.update("jax_disable_jit", True)


def make_single_fiber(
    n_elements: int, x0: tuple, xN: tuple, cell_shift: int
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
    return points, cells


def make_fabric(n_elements: list[int], X: list[tuple],DirichletBC_ends: list[tuple]):
    point_blocks = []
    cell_blocks = []

    bcs = []

    vertex_offset = 0

    for (n_el, x, bc) in zip(n_elements, X, DirichletBC_ends):
        points_i, cells_i = make_single_fiber(
            n_elements=n_el,
            x0=x[0],
            xN=x[1],
            cell_shift=vertex_offset,
        )

        point_blocks.append(points_i)
        cell_blocks.append(cells_i)
        bcs += [
            DirichletBC(bc_type=BCType.NODE, component=c, index=i, value=bc[c])
            for i in (vertex_offset + 0, vertex_offset + n_el)
            for c in (0, 1, 2)
        ]

        vertex_offset += points_i.shape[0]

    points = np.vstack(point_blocks)

    fiber_offsets = np.concatenate(
        [
            [0],
            np.cumsum([b.shape[0] for b in point_blocks])
        ]
    )
    # fiber_offsets = np.cumsum([b.shape[0] for b in point_blocks])
    fabric = VTMSFabric(
        name="test",
        material_ids=np.array([0,1]),
        diameters=np.array([0.1,0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, 3, fiber_offsets.shape[0]-1]),
    )
    return fabric,bcs

def run_orthogonalTows(
    n_elements: list[int],
    X: list[tuple],
    DirichletBC_ends: list[tuple],
    pseudoT:int,
    filename_base:str,
    contact_params: ContactParams,
):
    """ """
    fabric, bcs = make_fabric(n_elements=n_elements, X=X, DirichletBC_ends=DirichletBC_ends)
    dyn_bcs = []
    for ii in range(pseudoT):
        temp_bcs =deepcopy(bcs)
        for temp_bc,control_bc in zip(temp_bcs,bcs):
            temp_bc.value = control_bc.value*(ii+1)
        dyn_bcs.append(temp_bcs)

    d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_dist = d[d.nonzero()].min()
    E = 1e9
    A = (fabric.diameters[0]/2)**2*np.pi
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
            nonlinear_max_iter=20,
            linear_max_iter=500,
            # max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
        ),
        pseudotime_iters=pseudoT,
        filename_base=filename_base,
        blow_up_threshold=1e6,
        plot_convergence=False,
    )
    return u,fabric

args = {
    'n_elements':[40]*6,
    'X':[
        [[0.05,-1,0.05],[0.05,1,0.05]],
        [[0.05,-1,-0.05],[0.05,1,-0.05]],
        [[0.05+0.1*np.sqrt(3)/2,-1,0.0],[0.05+0.1*np.sqrt(3)/2,1,0.0]],
        [[-0.05-0.1*np.sqrt(3)/2,0,-1],[-0.05-0.1*np.sqrt(3)/2,0,1]],
        [[-0.05,0.05,-1],[-0.05,0.05,1]],
        [[-0.05,-0.05,-1],[-0.05,-0.05,1]],
    ],
    'DirichletBC_ends':[
        *[[-0.005,0,0]]*3,
        *[[ 0.005,0,0]]*3,
    ],
    'contact_params': ContactParams(
        self_adjacency_block    = 10000,
        contact_stiffness_model = __contact_stiffness_exponential,
        D_stiffness_to_E_ratio  = 0.25,
        contact_search_radius   = 0.5,
        M_to_D_ratio            = 1.25,
        M_stiffness_to_E_ratio  = 1.0/100.0
    ),
    'pseudoT':5,
    'filename_base':'TestContactParams/Exponential_Dirichlet',
}


run_orthogonalTows(**args)
# args['filename_base'] = 'ContactStiffnessModel/Linear_DirichletTest'
# args['contact_stiffness_model'] = contact_stiffness_linear
# ul_whole,fl_whole = run_orthogonalTows(**args)
# args['filename_base'] = 'ContactStiffnessModel/Piecewise_DirichletTest'
# args['contact_stiffness_model'] = contact_stiffness_piecewise_linear
# up_whole,fp_whole = run_orthogonalTows(**args)
# args['filename_base'] = 'ContactStiffnessModel/Exponential_DirichletTest'
# args['contact_stiffness_model'] = contact_stiffness_exponential
# ue_whole,fe_whole = run_orthogonalTows(**args)

# args['contact_search_radius'] = 0.25
# args['filename_base'] = 'ContactStiffnessModel/Linear_HalfS_DirichletTest'
# args['contact_stiffness_model'] = contact_stiffness_linear
# ul_half,fl_half = run_orthogonalTows(**args)
# args['filename_base'] = 'ContactStiffnessModel/Piecewise_HalfS_DirichletTest'
# args['contact_stiffness_model'] = contact_stiffness_piecewise_linear
# up_half,fp_half = run_orthogonalTows(**args)
# args['filename_base'] = 'ContactStiffnessModel/Exponential_HalfS_DirichletTest'
# args['contact_stiffness_model'] = contact_stiffness_exponential
# ue_half,fe_half = run_orthogonalTows(**args)

def get_min(fabric,i,j):
    fi = fabric.get_fiber_points(0,i)
    fj = fabric.get_fiber_points(0,j)
    D = np.linalg.norm(fi[None,:,:] - fj[:,None,:],axis=-1)
    return D[D.nonzero()].min()

def get_mins(fabric):
    n = fabric.get_n_fibers_in_bundle(0)
    M = []
    for i in range(n):
        for j in range(i+1,n):
            print(f"({i},{j}) - {get_min(fabric,i,j)}")
            M.append(get_min(fabric,i,j))
    return np.array(M).min()

# print(f"{get_mins(fl_whole):0.6f}")
# print(f"{get_mins(fp_whole):0.6f}")
# print(f"{get_mins(fe_whole):0.6f}")
# print(f"{get_mins(fl_half):0.6f}")
# print(f"{get_mins(fp_half):0.6f}")
# print(f"{get_mins(fe_half):0.6f}")
