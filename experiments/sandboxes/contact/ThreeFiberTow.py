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


def make_bundle(n_elements: list[int], X0: list[tuple], XN: list[tuple],NeumannForce):
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
        bcs += [
            NeumannBC(
                bc_type=BCType.NODE,
                component=0,
                index=vertex_offset + int(n_el / 2) + s,
                value=NeumannForce,
            )
            for s in (-1, 0, 1)
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
        bundle_offsets=np.array([0, fiber_offsets.shape[0]]),
    )
    return fabric,bcs

def run_threeFiberTow(
    n_elements: list[int],
    X0: list[tuple],
    XN: list[tuple],
    contact_search_radius: float,
    NeumannForce
):
    """ """
    fabric, bcs = make_bundle(n_elements=n_elements, X0=X0, XN=XN,NeumannForce=NeumannForce)
    write_vtk(fabric,get_output(filename="threeFiberTow_pre.vtk", subdir="contact"))

    d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_dist = d[d.nonzero()].min()
    E = 1e9
    A = (fabric.diameters[0]/2)**2*np.pi
    print(f"EA/N = {E*A/NeumannForce}")
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[VTMSFiberMaterial(id=0, E=E, A=A)],
        boundary_conditions=bcs,
        contact_search_radius=contact_search_radius,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=1000,
            linear_max_iter=500,
            max_linear_displacement=min(min_dist/2,fabric.diameters[0]/2),
        ),
        plot_convergence=True,
    )
    u = u.reshape((-1,3))
    fabric.points = fabric.points + u
    write_vtk(fabric,get_output(filename="threeFiberTow_post.vtk", subdir="contact"))

    return u,fabric

u,f = run_threeFiberTow(
    n_elements=[10, 10, 10],
    X0=[[0, 0, -1], [0.1, 0, -1], [0.5 * 0.1, np.sqrt(3) / 2 * 0.1, -1]],
    XN=[[0, 0, 1], [0.1, 0, 1], [0.5 * 0.1, np.sqrt(3) / 2 * 0.1, 1]],
    contact_search_radius=0.25,
    NeumannForce = 1E5
)

# from copy import deepcopy
# def janky_vtk_iter(filename: str, n_iterations: int):
#     f, _ = make_bundle(
#         n_elements=[10, 10, 10],
#         X0=[[0, 0, -1], [0.1, 0, -1], [0.5 * 0.1, np.sqrt(3) / 2 * 0.1, -1]],
#         XN=[[0, 0, 1], [0.1, 0, 1], [0.5 * 0.1, np.sqrt(3) / 2 * 0.1, 1]],
#         NeumannForce = 1E4
#     )
#     f_copy = deepcopy(f)
#     p = deepcopy(f_copy.points)
#     write_vtk(f_copy,get_output(f"u_{0}.vtk",subdir = f"contact/{filename}"))

#     for i in range(n_iterations+1):
#         u = np.load(f"output/contact/u_blowup_NEW_iter{i}.npy")
#         f_copy.points = p+u
#         write_vtk(f_copy,get_output(f"u_{i+1}.vtk",subdir = f"contact/{filename}"))

