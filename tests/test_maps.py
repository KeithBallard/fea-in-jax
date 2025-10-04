import meshio
from helper import *
import jax.experimental.sparse as jsparse

for mesh_size in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:

    mesh = meshio.read(get_mesh(f"polygon_mesh_{mesh_size}.vtk"))

    # Shape (N_e, N_n, N_x)
    points = np.array(mesh.points, dtype=np.float32)
    cells = np.array(mesh.cells[1].data, dtype=np.uint64)
    print("points", points.shape) #, points)
    print("cells", cells.shape) #, cells)

    A = mesh_to_sparse_assembly_map(points.shape[0], cells)
    print("A", A.shape)  #, A)

    x_n = mesh_to_jax(points, cells)
    print("x_n", x_n.shape)  # , x_n)

    x_n_reshaped = x_n.reshape(1, x_n.shape[0] * x_n.shape[1], x_n.shape[2])
    print("x_n_reshaped", x_n_reshaped.shape)  #, x_n_reshaped)

    n_cell_per_vert = get_n_cells_per_vert(points, cells)
    print("n_cell_per_vert", n_cell_per_vert.shape)  # , n_cell_per_vert)

    transformed_points = jsparse.bcsr_dot_general(
        A, x_n_reshaped, dimension_numbers=(((2,), (1,)), ((0,), (0,)))
    )
    transformed_points = transformed_points / n_cell_per_vert[np.newaxis, :, np.newaxis]
    print("transformed_points", transformed_points.shape)  # , transformed_points)

    # Make sure the coordinates are close to the original mesh
    assert jnp.isclose(jnp.array(points[:, 0:2]), transformed_points[0]).all()
    
