import pyvista as pv
import os

# Load the original mesh
script_directory = os.path.dirname(__file__)
os.chdir(script_directory)
mesh = pv.read("microscale_2D_r0.vtk")

# Decimate the mesh (target_reduction is the fraction of triangles to remove)
# For example, 0.5 means remove 50% of the triangles.
# We first extract the surface because PyVista's decimate only works on PolyData objects, not UnstructuredGrids.
poly_mesh = mesh.extract_surface()
coarse_mesh = poly_mesh.decimate(target_reduction=0.5)

# Save the coarsened mesh
coarse_mesh.save("microscale_2D_coarse.vtk")
