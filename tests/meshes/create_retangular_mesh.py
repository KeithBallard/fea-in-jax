import gmsh
import sys, os

# Change directories to the test/meshes dir
script_directory = os.path.dirname(__file__)
os.chdir(script_directory)

# --- 1. Initialize Gmsh ---
gmsh.initialize()
gmsh.model.add("rectangle")

# --- 2. Define Geometry ---

# We can define the rectangle's corners and the desired mesh size at those points.
# The 'lc' variable stands for "characteristic length," which is Gmsh's way of
# specifying the target mesh element size.
lc = 0.1
p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
p2 = gmsh.model.occ.addPoint(2, 0, 0, lc)
p3 = gmsh.model.occ.addPoint(2, 1, 0, lc)
p4 = gmsh.model.occ.addPoint(0, 1, 0, lc)

# Define the lines that form the rectangle's boundary.
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)

# Create a "Curve Loop" from the lines. A curve loop is a closed boundary.
cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

# Create a "Plane Surface" from the curve loop. This is the rectangular domain.
s = gmsh.model.occ.addPlaneSurface([cl])

# Synchronize the CAD model with the Gmsh model. This is a crucial step before meshing.
gmsh.model.occ.synchronize()

# --- 3. Generate Mesh ---

# Generate the 2D mesh. Gmsh will automatically use a triangular algorithm.
gmsh.model.mesh.generate(2) # The '2' indicates 2D meshing.

# --- 4. Save and Visualize ---

# Save the mesh to a file. The .msh format is native to Gmsh.
output_filename = "rectangle.vtk"
gmsh.write(output_filename)
print(f"Mesh saved to {output_filename}")

# Finalize the Gmsh API.
gmsh.finalize()