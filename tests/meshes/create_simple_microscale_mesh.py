import gmsh
import sys, os

# It's good practice to avoid changing the working directory if possible.
# Let's define the output path relative to the script's location.
# Note: Since __file__ is not defined in all environments (like interactive notebooks),
# we'll default to the current working directory if it's not found.
try:
    script_directory = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_directory = os.getcwd()
output_filename = os.path.join(script_directory, "simple_microscale_2D.vtk")

# --- 1. Initialize Gmsh ---
gmsh.initialize()
gmsh.model.add("simple_microscale_2D")

# --- 2. Define Geometry ---

# We can define the rectangle's corners and the desired mesh size at those points.
# The 'lc' variable stands for "characteristic length," which is Gmsh's way of
# specifying the target mesh element size.
lc = 1e-6 # Using a slightly finer mesh to better resolve the circle
domain_width = 1.5e-5
domain_height = 1.5e-5

# -- Define the outer rectangle --
p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
p2 = gmsh.model.occ.addPoint(domain_width, 0, 0, lc)
p3 = gmsh.model.occ.addPoint(domain_width, domain_height, 0, lc)
p4 = gmsh.model.occ.addPoint(0, domain_height, 0, lc)

l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)

# Create a "Curve Loop" for the outer rectangular boundary.
cl_outer = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

# --- Define the circular fiber in the middle ---

# Define the circle's parameters
center_x = domain_width / 2
center_y = domain_height / 2
radius = 5e-6

# To create a circle in Gmsh, we define its center point and points on its
# circumference, then connect them with arcs.
p_center = gmsh.model.occ.addPoint(center_x, center_y, 0, lc)
p_left   = gmsh.model.occ.addPoint(center_x - radius, center_y, 0, lc)
p_top    = gmsh.model.occ.addPoint(center_x, center_y + radius, 0, lc)
p_right  = gmsh.model.occ.addPoint(center_x + radius, center_y, 0, lc)
p_bottom = gmsh.model.occ.addPoint(center_x, center_y - radius, 0, lc)

# Create the circular arcs using the center point.
# gmsh.model.occ.addCircleArc(start_point_tag, center_point_tag, end_point_tag)
arc1 = gmsh.model.occ.addCircleArc(p_left, p_center, p_top)
arc2 = gmsh.model.occ.addCircleArc(p_top, p_center, p_right)
arc3 = gmsh.model.occ.addCircleArc(p_right, p_center, p_bottom)
arc4 = gmsh.model.occ.addCircleArc(p_bottom, p_center, p_left)

# Create a "Curve Loop" for the inner circular boundary.
cl_fiber = gmsh.model.occ.addCurveLoop([arc1, arc2, arc3, arc4])

# --- MODIFIED: Create the two Plane Surfaces (outer and inner) ---

# 1. Create the outer surface (the square with a circular hole).
#    The first curve loop is the outer boundary, and the second is the hole.
s_outer = gmsh.model.occ.addPlaneSurface([cl_outer, cl_fiber])

# 2. Create the inner surface (the filled circle).
#    This uses only the circular curve loop as its boundary.
s_inner = gmsh.model.occ.addPlaneSurface([cl_fiber])

# Synchronize the CAD model with the Gmsh model. This is a crucial step.
gmsh.model.occ.synchronize()


# --- 3. Add Physical Groups (Good Practice!) ---
# This step allows us to label the different regions of the mesh.

# Add a physical group for the outer surface (the "matrix")
# Arguments: dimension (2 for surface), entity tags ([s_outer]), group tag (1)
gmsh.model.addPhysicalGroup(2, [s_outer], 1)
gmsh.model.setPhysicalName(2, 1, "Matrix")

# Add a physical group for the inner surface (the "fiber")
gmsh.model.addPhysicalGroup(2, [s_inner], 2)
gmsh.model.setPhysicalName(2, 2, "Fiber")


# --- 4. Generate Mesh ---

# Generate the 2D mesh. Gmsh will mesh both surfaces.
gmsh.model.mesh.generate(2) # The '2' indicates 2D meshing.

# --- 5. Save and Visualize ---

# Save the mesh to a file. The .vtk format is viewable in ParaView/VTK.
gmsh.write(output_filename)
print(f"Mesh saved to {output_filename}")

# You can uncomment the following line to launch the Gmsh GUI to see the result
# if gmsh.fltk.isAvailable():
#     gmsh.fltk.run()

# Finalize the Gmsh API.
gmsh.finalize()