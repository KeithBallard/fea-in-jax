import pygmsh
import meshio

# Mesh of a polygon
for mesh_size in [0.5, 0.1,0.07,0.065,0.06,0.055, 0.05,0.045,0.04,0.035,0.03, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001]:
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0.0, 0.0],
                [1.0, -0.2],
                [1.1, 1.2],
                [0.1, 0.7],
            ],
            mesh_size=mesh_size,
        )
        mesh = geom.generate_mesh()

    mesh.write(f"tests/meshes/polygon_mesh_{mesh_size}.vtk")

# Test reading the meshes
for mesh_size in [0.05, 0.01, 0.005, 0.001]:
    mesh = meshio.read(f"tests/meshes/polygon_mesh_{mesh_size}.vtk")