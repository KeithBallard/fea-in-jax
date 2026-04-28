import numpy as np
import pyvista
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, plot
import dolfinx.fem.petsc

l = 5.0
h1 = 3.33
h2 = 5.33
h3 = 6.0

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
geom = gmsh.model.geo

left = geom.add_point(0.0, 0.0, 0.0)
right = geom.add_point(6 * l, 0.0, 0.0)
bottom_points = [
        geom.add_point(l, 0.0, 0.0),
        geom.add_point(2 * l, 0.0, 0.0),
        geom.add_point(3 * l, 0.0, 0.0),
        geom.add_point(4 * l, 0.0, 0.0),
        geom.add_point(5 * l, 0.0, 0.0),
]
top_points = [
        geom.add_point(l, h1, 0.0),
        geom.add_point(2 * l, h2, 0.0),
        geom.add_point(3 * l, h3, 0.0),
        geom.add_point(4 * l, h2, 0.0),
        geom.add_point(5 * l, h1, 0.0),
]
ext_bottom_points = [left] + bottom_points + [right]
ext_top_points = [left] + top_points + [right]

bottom_lines = [
        geom.add_line(ext_bottom_points[i], ext_bottom_points[i + 1])
        for i in range(len(ext_bottom_points) - 1)
]

top_lines = [
        geom.add_line(ext_top_points[i], ext_top_points[i + 1])
        for i in range(len(ext_top_points) - 1)
]

vertical_lines = [geom.add_line(p1, p2) for p1, p2 in zip(bottom_points, top_points)]

left_diagonal_lines = [
        geom.add_line(top_points[i], bottom_points[i + 1])
        for i in range(len(bottom_points) - 1)
]
right_diagonal_lines = [
        geom.add_line(bottom_points[i], top_points[i + 1])
        for i in range(len(bottom_points) - 1)
]
lines = (
        bottom_lines
        + top_lines
        + vertical_lines
        + right_diagonal_lines
        + left_diagonal_lines
)
geom.synchronize()

for l in lines:
    gmsh.model.mesh.set_transfinite_curve(l, 2)
gmsh.model.add_physical_group(0, bottom_points, 1)
gmsh.model.add_physical_group(0, [left] + [right], 2)
gmsh.model.add_physical_group(1, bottom_lines, 1)
gmsh.model.add_physical_group(1, top_lines
                                  + vertical_lines
                                  + right_diagonal_lines
                                  + left_diagonal_lines, 2)
gmsh.model.mesh.generate(dim=1)
gmsh.write("truss.msh")

domain, markers, facets = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)[:3]

gmsh.finalize()

gdim = domain.geometry.dim
tdim = domain.topology.dim

print(f"Geometrical dimension = {gdim}")
print(f"Topological dimension = {tdim}")

V = fem.functionspace(domain, ("CG", 1, (gdim,)))
Vv = fem.functionspace(domain, ("DG", 0, (gdim,)))

dx_dX = ufl.Jacobian(domain)[:, 0]
t_ufl = dx_dX / ufl.sqrt(ufl.inner(dx_dX, dx_dX))
t = fem.Function(Vv, name="Tangent_vector")
t.interpolate(fem.Expression(t_ufl, Vv.element.interpolation_points))

F = fem.Function(V)
fdim = tdim - 1
load_dofs = fem.locate_dofs_topological(V.sub(1), fdim, facets.find(1))
F.x.array[load_dofs] = -1

bc_dofs = fem.locate_dofs_topological(V, fdim, facets.find(2))
bcs = [fem.dirichletbc(np.zeros((gdim,)), bc_dofs, V)]

def plot_truss_before():
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()
    
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    F_3D = np.zeros((u_geometry.shape[0], 3))
    F_3D[:, :2] = F.x.array.reshape(-1, 2)
    u_grid.point_data["F"] = F_3D
    glyphs = u_grid.glyph(
            orient="F",
            factor=3.0,
    )
    plotter.add_mesh(glyphs, color="black")
    
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain))
    plotter.add_mesh(grid, show_edges=True, line_width=10, color="royalblue")
    t_3D = np.zeros((grid.n_cells, 3))
    t_3D[:, :2] = t.x.array.reshape(-1, 2)
    grid.cell_data["t"] = t_3D
    grid.set_active_vectors("t")
    arrow = pyvista.Arrow(
            start=(-0.5, 0, 0),
    )
    glyphs = grid.glyph(orient="t", factor=3.0, geom=arrow)
    
    plotter.add_mesh(glyphs, color="darkred")
    plotter.show_axes()
    plotter.view_xy()
    plotter.zoom_camera(1.3)
    plotter.show()
    plotter.screenshot("truss_before.png")

# Define the variational problem
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

E = fem.Constant(domain, 200e3)
S = fem.Constant(domain, 1.0)

def strain(u):
    return ufl.dot(ufl.dot(ufl.grad(u), t_ufl), t_ufl)

def normal_force(u):
    return E * S * strain(u)

dx = ufl.Measure("dx", subdomain_data=markers)
a_form = fem.form(ufl.inner(normal_force(du), strain(u_)) *dx)

# Concentrated Loadings
F0 = fem.Constant(domain, np.zeros((gdim,)))
L_form = fem.form(ufl.dot(F0, u_) * dx)

A = fem.petsc.assemble_matrix(a_form, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L_form)
with b.localForm() as loc:
        loc.set(0.0)

# Example: apply force at dof i
for dof in load_dofs:
    b.setValue(dof, -1)
b.assemble()

# b.array[:] = F.x.array[:]
# fem.petsc.apply_lifting(b, [a_form], bcs=[bcs])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
# fem.petsc.set_bc(b, bcs)

# Solving the problem
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

solver.solve(b, u.x.petsc_vec)
u.x.scatter_forward()

V0 = fem.functionspace(domain, ("DG", 0, ()))
N_exp = fem.Expression(normal_force(u), V0.element.interpolation_points)
N = fem.Function(V0, name="Normal_force")
N.interpolate(N_exp)

def plot_truss_after():
    plotter = pyvista.Plotter()
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_3D = np.zeros((u_geometry.shape[0], 3))
    u_3D[:, :2] = u.x.array.reshape(-1, 2)
    u_3D[:, 2] = (
            1e-3  # slightly offset to avoid overlap in xy view with underlying undeformed mesh
    )
    u_grid.point_data["Deflection"] = u_3D
    u_grid.set_active_vectors("Deflection")
    warped = u_grid.warp_by_vector("Deflection", factor=2000.0)
    warped.cell_data["Normal force"] = N.x.array
    plotter.add_mesh(
            u_grid, show_edges=True, line_width=10, color="royalblue", opacity=0.25
    )
    
    Nmax = max(abs(N.x.array))
    plotter.add_mesh(
            warped,
            show_scalar_bar=True,
            scalars="Normal force",
            render_lines_as_tubes=True,
            style="wireframe",
            line_width=10,
            opacity=1,
            cmap="bwr",
            clim=[-Nmax, Nmax],
    )
    plotter.view_xy()
    plotter.zoom_camera(1.3)
    plotter.show()
    plotter.screenshot("truss_after.png")

dof_coords = u.function_space.tabulate_dof_coordinates()
vals = u.x.array.reshape((-1, 2))

# print('-'*20)
# print('  coords   |   vals   ')
# for x, v in zip(dof_coords, vals):
#         print(x, v)
print('-'*77)
print(f"{'coordinates':^26}|{'displacements':^48}")
print('-'*77)
for x, v in zip(dof_coords, vals):
    # Format each coordinate and value to 6 decimal places
    coord_str = "[" + " ".join(f"{xi:7.1f}" for xi in x) + "]"
    val_str = "[" + " ".join(f"{vi: .16e}" for vi in v) + "]"
    print(f"{coord_str:>15} | {val_str:>48}")
