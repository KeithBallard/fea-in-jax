from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
# jax.config.update("jax_disable_jit", True)


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


def make_bundle(n_elements: list[int], X: list[tuple],DirichletBC_ends: list[tuple]):
    point_blocks = []
    cell_blocks = []
    point_id_blocks = []
    cell_id_blocks = []

    bcs = []

    vertex_offset = 0

    for fiber_id, (n_el, x, bc) in enumerate(zip(n_elements, X, DirichletBC_ends)):
        points_i, cells_i, point_ids_i, cell_ids_i = make_single_fiber(
            n_elements=n_el,
            x0=x[0],
            xN=x[1],
            fiber_id=fiber_id,
            cell_shift=vertex_offset,
        )

        point_blocks.append(points_i)
        cell_blocks.append(cells_i)
        point_id_blocks.append(point_ids_i)
        cell_id_blocks.append(cell_ids_i)
        bcs += [
            DirichletBC(bc_type=BCType.NODE, component=c, index=i, value=bc[c])
            for i in (vertex_offset + 0, vertex_offset + n_el)
            for c in (0, 1, 2)
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

def run_orthogonalTows(
    n_elements: list[int],
    X: list[tuple],
    contact_search_radius: float,
    DirichletBC_ends: list[tuple],
    pseudoT:int,
    filename_base:str,
):
    """ """
    fabric, bcs = make_bundle(n_elements=n_elements, X=X, DirichletBC_ends=DirichletBC_ends)
    # return fabric
    if filename_base is not None:
        write_vtk(fabric,get_output(filename=f"{filename_base}_0.vtk", subdir="contact"))

    d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_dist = d[d.nonzero()].min()
    E = 1e9
    A = (fabric.diameters[0]/2)**2*np.pi
    for i in range(pseudoT):
        print(f"\n \n   pseudo-time i = {i+1}\n \n")
        u, _, _ = solve_fiber_mechanics_bvp(
            fabric=fabric,
            materials=[VTMSFiberMaterial(id=0, E=E, A=A)],
            boundary_conditions=bcs,
            contact_search_radius=contact_search_radius,
            solver_options=SolverOptions(
                # linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
                # linear_precond_type=PreconditionerType.JACOBI,
                nonlinear_max_iter=20,
                linear_max_iter=500,
                # max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
            ),
            plot_convergence=False,
        )
        u = u.reshape((-1,3))
        print(f"\nmax(||u||) = {np.linalg.norm(u,axis=1).max()}\n")
        fabric.points = fabric.points + np.array(u)
        if filename_base is not None:
            write_vtk(fabric,get_output(filename=f"{filename_base}_{i+1}.vtk", subdir="contact"))
        if jnp.isnan(u).any() or jnp.isinf(u).any() or np.linalg.norm(u,axis=1).max()>1e5:
            raise RuntimeError("Nonlinear solve diverged: displacement magnitude exceeded threshold")

    return u,fabric

# u,f = run_orthogonalTows(
#     n_elements=[10]*6,
#     X = [
#         [[0.05,-1,0.05],[0.05,1,0.05]],
#         [[0.05,-1,-0.05],[0.05,1,-0.05]],
#         [[0.05+0.1*np.sqrt(3)/2,-1,0.0],[0.05+0.1*np.sqrt(3)/2,1,0.0]],
#         [[-0.05-0.1*np.sqrt(3)/2,0,-1],[-0.05-0.1*np.sqrt(3)/2,0,1]],
#         [[-0.05,0.05,-1],[-0.05,0.05,1]],
#         [[-0.05,-0.05,-1],[-0.05,-0.05,1]],
#     ],
#     DirichletBC_ends = [
#         *[[-0.005,0,0]]*3,
#         *[[ 0.005,0,0]]*3,
#     ],
#     contact_search_radius = 0.25,
#     pseudoT = 80,
#     filename_base="OrthogonalTows_CG"
# )
args = {
    'n_elements':[10]*6,
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
    'contact_search_radius':0.25,
    'pseudoT':20,
    'filename_base':None,
}

