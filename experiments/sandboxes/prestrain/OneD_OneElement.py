from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
# jax.config.update("jax_disable_jit", True)


def make_single_fiber(
    n_elements: int, x0: tuple, xN: tuple, cell_shift: int
):
    points = np.vstack(
        [
            np.linspace(x0[i], xN[i], n_elements + 1) for i in range(len(x0))
        ]
    ).T
    cells = np.array(
        [[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)],
        dtype=np.uint64,
    )
    return points, cells


def make_fabric(n_elements: list[int], X: list[tuple]):
    point_blocks = []
    cell_blocks = []

    vertex_offset = 0

    for (n_el, x) in zip(n_elements, X):
        points_i, cells_i = make_single_fiber(
            n_elements=n_el,
            x0=x[0],
            xN=x[1],
            cell_shift=vertex_offset,
        )

        point_blocks.append(points_i)
        cell_blocks.append(cells_i)

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
        material_ids=np.array([0]),
        diameters=np.array([0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, fiber_offsets.shape[0]-1]),
    )
    return fabric

def run_1D(
    n_elements: list[int],
    X: list[tuple],
    pseudoT:int,
    filename_base:str,
    contact_params: ContactParams,
    pre_strain:float,
):
    """ """
    fabric = make_fabric(n_elements=n_elements, X=X)
    dyn_bcs = [[DirichletBC(index=0, component=0, value=0, bc_type=BCType.NODE)]]

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
            nonlinear_max_iter=20,
            linear_max_iter=500,
            # max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
        ),
        pseudotime_iters=pseudoT,
        filename_base=filename_base,
        blow_up_threshold=1e6,
        plot_convergence=False,
        pre_strain=pre_strain,
    )
    return u,fabric

args = {
    'n_elements':[1],
    'X':[[[0],[1]]],
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
}
