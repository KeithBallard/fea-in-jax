from fe_jax.boundary_conditions import NeumannBC
from fe_jax import contact
from helper import *
import pytest
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import jax.extend

pytestmark = pytest.mark.truss

if __name__ == "__main__":
    print(jax.extend.backend.get_backend().platform)
# from jax_smi import initialise_tracking
# initialise_tracking()

def make_fiber(n_elements: int, x0: tuple, xN: tuple, fiber_id: int, cell_shift: int):
    points=np.vstack((
        np.linspace(x0[0],xN[0],n_elements+1),
        np.linspace(x0[1],xN[1],n_elements+1),
        np.linspace(x0[2],xN[2],n_elements+1),
    )).T
    cells = np.array([[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)], dtype=np.uint64)
    fiber_ids = np.array([[fiber_id]]*len(points))
    cell_ids = np.array([[fiber_id]]*len(cells))
    return points,cells,fiber_ids,cell_ids

def run_two_fiber_contact(n_elements: list[int], X0: list[tuple], XN: list[tuple]):
    """
    This test case assums ALL rods start at (0,0,0),
    this makes the stretch/compression easy to apply.
    """
    point_blocks = []
    cell_blocks = []
    point_id_blocks = []
    cell_id_blocks = []

    vertex_offset = 0

    for fiber_id, (n_el, x0, xN) in enumerate(zip(n_elements, X0, XN)):
        points_i, cells_i, point_ids_i, cell_ids_i = make_fiber(
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

        vertex_offset += points_i.shape[0]

    points = np.vstack(point_blocks)
    cells = np.vstack(cell_blocks)
    point_ids = np.vstack(point_id_blocks)
    cell_ids = np.vstack(cell_id_blocks)

    contacts = contact.contact_batch([points[point_ids.reshape(-1) == i,:] for i in range(2)],0.5)
    contact_point_blocks = []
    contact_cell_blocks = []
    contact_point_id_blocks = []
    contact_cell_id_blocks = []
    for con in contacts:
        points_i, cells_i, point_ids_i, cell_ids_i = make_fiber(
            n_elements=1,
            x0=con.x_i,
            xN=con.x_j,
            fiber_id=cell_ids.max()+1,
            cell_shift=vertex_offset,
        )

        contact_point_blocks.append(points_i)
        contact_cell_blocks.append(cells_i)
        contact_point_id_blocks.append(point_ids_i)
        contact_cell_id_blocks.append(cell_ids_i)

        vertex_offset += points_i.shape[0]

    points    = np.vstack((points, np.vstack(contact_point_blocks)))
    cells     = np.vstack((cells, np.vstack(contact_cell_blocks)))
    point_ids = np.vstack((point_ids, np.vstack(contact_point_id_blocks)))
    cell_ids  = np.vstack((cell_ids, np.vstack(contact_cell_id_blocks)))

    return points,cells,point_ids,cell_ids

def write_two_fiber_mesh(points, cells, point_ids, cell_ids, filename):
    """
    Write a 3D truss mesh for ParaView.

    Parameters
    ----------
    points : (V, 3) array
        Vertex coordinates.
    cells : (E, 2) array
        Line connectivity.
    point_ids : (V, 1) array
        Per-point fiber ids.
    cell_ids : (E, 1) array
        Per-cell fiber ids.
    filename : str
        Output mesh filename, e.g. "two_fibers_init.vtk"
    """
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=np.float64),
        cells=[
            ("line", np.asarray(cells, dtype=np.uint64)),
        ],
        point_data={
            "fiber_id": np.asarray(point_ids, dtype=np.int64).reshape(-1),
        },
        cell_data={
            "fiber_id": [
                np.asarray(cell_ids, dtype=np.int64).reshape(-1),
            ],
        },
    )
    mesh.write(get_output(filename))

