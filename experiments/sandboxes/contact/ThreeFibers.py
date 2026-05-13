from fe_jax.boundary_conditions import NeumannBC
from fe_jax import contact
from helper import *
import pytest
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import jax.extend


if __name__ == "__main__":
    print(jax.extend.backend.get_backend().platform)
# from jax_smi import initialise_tracking
# initialise_tracking()

def sci_to_latex(val):
    """
    Convert a float to LaTeX scientific notation: a \\times 10^{b}
    """
    s = f"{val:.3e}"           # e.g., -1.234e-05
    mantissa, exp = s.split('e')
    exp = int(exp)             # remove leading zeros like e-05 -> -5

    # Clean mantissa (remove trailing zeros if desired)
    mantissa = f"{float(mantissa):.3f}".rstrip('0').rstrip('.')

    return f"{mantissa} \\cdot 10^{{{exp}}}"


def export_to_latex_table(points, u_truss, filename="table.txt"):
    with open(filename, "w") as f:
        f.write("\\begin{tabular}{ccc|ccc}\n")
        f.write("\\hline\n")

        f.write("\\multicolumn{3}{c|}{\\textbf{points}} & "
                "\\multicolumn{3}{c}{\\textbf{displacement}} \\\\\n")

        f.write("\\hline\n")

        for p, v in zip(points, u_truss):
            # coord_vals = [f"${xi:0.3g}$" for xi in p]
            coord_vals = [f"${f'{xi:.3f}'.rstrip('0').rstrip('.')}$" for xi in p]
            val_vals   = [f"${sci_to_latex(vi)}$" for vi in v]

            row = " & ".join(coord_vals + val_vals)
            f.write(row + " \\\\\n")

        f.write("\\end{tabular}\n")

def write_two_fiber_mesh_with_deformation_opt(
    points,
    cells,
    point_ids,
    cell_ids,
    filename,
    u_truss=None,
):
    """
    Write a VTK mesh for the fibers.

    If u_truss is None, write only the undeformed geometry.
    If u_truss is provided, write both undeformed and deformed geometry.

    ParaView can color by:
      - point_data["fiber_id"]
      - cell_data["fiber_id"]
      - cell_data["state"] if deformation is written
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.uint64)
    point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)

    if u_truss is None:
        mesh = meshio.Mesh(
            points=points,
            cells=[("line", cells)],
            point_data={
                "fiber_id": point_ids,
            },
            cell_data={
                "fiber_id": [cell_ids],
            },
        )
    else:
        u_truss = np.asarray(u_truss, dtype=np.float64).reshape(points.shape)
        V = points.shape[0]
        E = cells.shape[0]

        points_vis = np.vstack((points, points + u_truss))

        mesh = meshio.Mesh(
            points=points_vis,
            cells=[
                ("line", cells),
                ("line", cells + V),
            ],
            point_data={
                "fiber_id": np.concatenate((point_ids, point_ids)),
            },
            cell_data={
                "fiber_id": [
                    cell_ids,
                    cell_ids,
                ],
                "state": [
                    np.zeros(E, dtype=np.int64),
                    np.ones(E, dtype=np.int64),
                ],
            },
        )

    mesh.write(filename)
def write_two_fiber_mesh_with_deformation(points, cells, point_ids, cell_ids, u_truss, filename):
    """
    Write a single VTK mesh containing both:
      - undeformed geometry: points
      - deformed geometry: points + u_truss

    ParaView can color by:
      - cell_data["fiber_id"] to distinguish fibers
      - cell_data["state"] to distinguish before/after
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.uint64)
    point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    u_truss = np.asarray(u_truss, dtype=np.float64).reshape(points.shape)

    V = points.shape[0]
    E = cells.shape[0]

    points_vis = np.vstack((points, points + u_truss))

    mesh = meshio.Mesh(
        points=points_vis,
        cells=[
            ("line", cells),
            ("line", cells + V),
        ],
        point_data={
            "fiber_id": np.concatenate((point_ids, point_ids)),
        },
        cell_data={
            "fiber_id": [
                cell_ids,
                cell_ids,
            ],
            "state": [
                np.zeros(E, dtype=np.int64),  # before
                np.ones(E, dtype=np.int64),   # after
            ],
        },
    )

    mesh.write(filename)

def make_single_fiber(n_elements: int, x0: tuple, xN: tuple, fiber_id: int, cell_shift: int):
    points=np.vstack((
        np.linspace(x0[0],xN[0],n_elements+1),
        np.linspace(x0[1],xN[1],n_elements+1),
        np.linspace(x0[2],xN[2],n_elements+1),
    )).T
    cells = np.array([[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)], dtype=np.uint64)
    fiber_ids = np.array([[fiber_id]]*len(points))
    cell_ids = np.array([[fiber_id]]*len(cells))
    return points,cells,fiber_ids,cell_ids

def make_fibers(n_elements: list[int], X0: list[tuple], XN: list[tuple]):
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
            DirichletBC(bc_type=BCType.NODE, component=c, index=i, value=0.0) for c in (0,1,2) for i in (vertex_offset+0,vertex_offset+n_el)
        ]
        bcs += [
            NeumannBC(bc_type=BCType.NODE, component=0, index=vertex_offset + int(n_el/2) + s, value=1E6) for s in (-1,0,1)
        ]

        vertex_offset += points_i.shape[0]

    points = np.vstack(point_blocks)
    cells = np.vstack(cell_blocks)
    point_ids = np.vstack(point_id_blocks).reshape(-1)
    cell_ids = np.vstack(cell_id_blocks).reshape(-1)

    return points, cells, point_ids, cell_ids, bcs

def visualizeContacts(points,point_ids,cells,cell_ids,search_radius,filename):
    capacity = contact.count_initial_contacts(
        points = points,
        point_fiber_ids = point_ids,
        adjacency_block = 50,
        radius = search_radius
    )
    print(f"Initially, {int(capacity)} contacts found with search radius {search_radius}")
    contact_cells,_ = contact.contact_batch(
        points = points,
        point_fiber_ids = point_ids,
        capacity=int(capacity),
        adjacency_block = 50,
        radius = search_radius
    )

    cells = np.vstack((cells, contact_cells))
    cells = np.array(cells,dtype=np.int64)
    cell_ids = np.hstack((cell_ids,(np.max(cell_ids)+1)*np.ones(contact_cells.shape[0])))
    cell_ids = np.array(cell_ids,dtype=np.int64)
    write_two_fiber_mesh_with_deformation_opt(
        points=points,
        cells=cells,
        point_ids=point_ids,
        cell_ids=cell_ids,
        filename=get_output(filename + "_init.vtk"),
    )


def run_threeFiberTow(n_elements: list[int], X0: list[tuple], XN: list[tuple],search_radius: float):
    """
    """
    points, cells, point_ids, cell_ids, bcs = make_fibers(
        n_elements=n_elements,
        X0=X0,
        XN=XN
    )
    visualizeContacts(
        points=points,
        point_ids=point_ids,
        cells=cells,
        cell_ids=cell_ids,
        search_radius=search_radius,
        filename = "threeFiberTow"
    )
    # search_radius = 0.5
    # contact_cells,_ = contact.contact_batch(
    #     points = points,
    #     point_fiber_ids = point_ids,
    #     n_contact=21,
    #     adjacency_block = 50,
    #     radius = search_radius)

    # cells = np.vstack((cells, contact_cells))
    # cells = np.array(cells,dtype=np.int64)
    # cell_ids = np.hstack((cell_ids,(np.max(cell_ids)+1)*np.ones(contact_cells.shape[0])))
    # cell_ids = np.array(cell_ids,dtype=np.int64)


    # Sizes of arrays
    U = 3  # number of solution components
    V = points.shape[0]  # number of vertices
    E = cells.shape[0]  # number of elements
    F = V * U  # number of DoFs
    fe_type = FiniteElementType(
        cell_type=CellType.interval,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points

    print("# DoFs = ", F)

    # Set material properties
    matrix_mat_params_contact = jnp.array([4e9,1,search_radius])  # E_max, A, R

    # Set boundary conditions.
    # The displacement is in the direciton of the bar, so this should be the same as a 1D displacement.
    # bcs = (
    #     [
    #         DirichletBC(bc_type=BCType.NODE, component=0, index=0         , value=-0.1),
    #         DirichletBC(bc_type=BCType.NODE, component=1, index=0         , value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=2, index=0         , value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=0, index=n_elements[0], value=-0.1),
    #         DirichletBC(bc_type=BCType.NODE, component=1, index=n_elements[0], value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=2, index=n_elements[0], value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=0, index=n_elements[0]+1, value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=1, index=n_elements[0]+1, value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=2, index=n_elements[0]+1, value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=0, index=n_elements[0]+n_elements[1]+1, value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=1, index=n_elements[0]+n_elements[1]+1, value=0.0),
    #         DirichletBC(bc_type=BCType.NODE, component=2, index=n_elements[0]+n_elements[1]+1, value=0.0),
    #     ]
    # )

    # Example using the truss elements
    element_batches_truss = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=3,
            connectivity_en=cells,
            constitutive_model=elastic_truss,
            material_params=matrix_mat_params,
        )
    ]
    contact_config = contact.ContactPreprocessConfig(
        vertices_fiber_ids = point_ids,
        radius = search_radius,
        self_adjacency_block = 3,
        material_params = matrix_mat_params_contact,
        fe_type = fe_type,
        constitutive_model = elastic_contact_truss
    )

    u_truss, residual_truss, element_batches_truss = solve_bvp(
        element_residual_func=linear_truss_residual,
        vertices_vd=points,
        element_batches=element_batches_truss,
        boundary_conditions=bcs,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=50,
            linear_max_iter=50,
        ),
        plot_convergence=False,
        contact_config=contact_config
    )
    u_truss = u_truss.reshape((-1,3))
    if not (np.isnan(u_truss).any() or np.isinf(u_truss).any()):
        write_two_fiber_mesh_with_deformation(
            points=points,
            cells=cells,
            point_ids=point_ids,
            cell_ids=cell_ids,
            u_truss=u_truss,
            filename=get_output("threeFiberTow_preprocess_contact.vtk"),
        )
        #     ],
    # return u_truss, points, cells

# Exampe Use Case
# run_truss_3D_bar([10,10],[(0.1,0,-1),(0,-1,0)],[(0.1,0,1),(0,1,0)],0.5)
