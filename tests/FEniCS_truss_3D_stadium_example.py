import numpy as np
import pyvista
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, plot
import dolfinx.fem.petsc


SUPPORT_TAG = 1
ROOF_LOAD_TAG = 2
TRUSS_TAG = 1

E_VALUE = 200e3
AREA_VALUE = 1.0
ROOF_LOAD_VECTOR = np.array([0.05, 0.15, -2.0])


def make_stadium_truss_geometry():
    n_stations = 6
    x = np.linspace(0.0, 30.0, n_stations)
    roof_z = np.array([3.0, 5.0, 6.2, 6.2, 5.0, 3.0])

    bottom_left = np.column_stack((x, -5.0 * np.ones_like(x), np.zeros_like(x)))
    bottom_right = np.column_stack((x, 5.0 * np.ones_like(x), np.zeros_like(x)))
    roof_left = np.column_stack((x, -7.0 * np.ones_like(x), roof_z))
    roof_right = np.column_stack((x, 7.0 * np.ones_like(x), roof_z))

    points = np.vstack((bottom_left, bottom_right, roof_left, roof_right))

    bl = np.arange(0, n_stations)
    br = np.arange(n_stations, 2 * n_stations)
    tl = np.arange(2 * n_stations, 3 * n_stations)
    tr = np.arange(3 * n_stations, 4 * n_stations)

    node_names = (
        [f"bottom_left_{i}" for i in range(n_stations)]
        + [f"bottom_right_{i}" for i in range(n_stations)]
        + [f"roof_left_{i}" for i in range(n_stations)]
        + [f"roof_right_{i}" for i in range(n_stations)]
    )

    cells = []

    def add_member(a, b):
        cells.append((int(a), int(b)))

    for i in range(n_stations - 1):
        add_member(bl[i], bl[i + 1])
        add_member(br[i], br[i + 1])
        add_member(tl[i], tl[i + 1])
        add_member(tr[i], tr[i + 1])

        add_member(bl[i], tl[i + 1])
        add_member(tl[i], bl[i + 1])
        add_member(br[i], tr[i + 1])
        add_member(tr[i], br[i + 1])

        add_member(tl[i], tr[i + 1])
        add_member(tr[i], tl[i + 1])
        add_member(bl[i], br[i + 1])
        add_member(br[i], bl[i + 1])

    for i in range(n_stations):
        add_member(bl[i], br[i])
        add_member(tl[i], tr[i])
        add_member(bl[i], tl[i])
        add_member(br[i], tr[i])

    support_nodes = [bl[0], br[0], bl[-1], br[-1]]
    roof_load_nodes = list(tl[1:-1]) + list(tr[1:-1])

    return (
        points,
        np.array(cells, dtype=np.int64),
        node_names,
        np.array(support_nodes, dtype=np.int64),
        np.array(roof_load_nodes, dtype=np.int64),
    )


points, cells, node_names, support_nodes, roof_load_nodes = make_stadium_truss_geometry()

gmsh.initialize()
gmsh.model.add("stadium_truss_3D")
gmsh.option.setNumber("General.Terminal", 0)
geom = gmsh.model.geo

point_tags = [geom.add_point(*point) for point in points]
line_tags = [geom.add_line(point_tags[a], point_tags[b]) for a, b in cells]
geom.synchronize()

for line in line_tags:
    gmsh.model.mesh.set_transfinite_curve(line, 2)

gmsh.model.add_physical_group(
    0, [point_tags[i] for i in support_nodes], SUPPORT_TAG
)
gmsh.model.set_physical_name(0, SUPPORT_TAG, "fixed_supports")
gmsh.model.add_physical_group(
    0, [point_tags[i] for i in roof_load_nodes], ROOF_LOAD_TAG
)
gmsh.model.set_physical_name(0, ROOF_LOAD_TAG, "loaded_roof_nodes")
gmsh.model.add_physical_group(1, line_tags, TRUSS_TAG)
gmsh.model.set_physical_name(1, TRUSS_TAG, "truss_members")

gmsh.model.mesh.generate(dim=1)
gmsh.write("stadium_truss_3D.msh")

domain, markers, facets = io.gmsh.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=3
)[:3]
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

for component, value in enumerate(ROOF_LOAD_VECTOR):
    load_dofs = fem.locate_dofs_topological(
        V.sub(component), fdim, facets.find(ROOF_LOAD_TAG)
    )
    F.x.array[load_dofs] = value

bc_dofs = fem.locate_dofs_topological(V, fdim, facets.find(SUPPORT_TAG))
bcs = [fem.dirichletbc(np.zeros((gdim,)), bc_dofs, V)]


def plot_truss_before():
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["F"] = F.x.array.reshape(-1, gdim)
    glyphs = u_grid.glyph(orient="F", factor=2.0)
    plotter.add_mesh(glyphs, color="black")

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain))
    plotter.add_mesh(grid, show_edges=True, line_width=4, color="royalblue")
    grid.cell_data["t"] = t.x.array.reshape(-1, gdim)
    grid.set_active_vectors("t")
    arrow = pyvista.Arrow(start=(-0.5, 0.0, 0.0))
    glyphs = grid.glyph(orient="t", factor=2.0, geom=arrow)

    plotter.add_mesh(glyphs, color="darkred")
    plotter.show_axes()
    plotter.view_isometric()
    plotter.show()
    plotter.screenshot("stadium_truss_3D_before.png")


du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

E = fem.Constant(domain, E_VALUE)
S = fem.Constant(domain, AREA_VALUE)


def strain(v):
    return ufl.dot(ufl.dot(ufl.grad(v), t_ufl), t_ufl)


def normal_force(v):
    return E * S * strain(v)


dx = ufl.Measure("dx", subdomain_data=markers)
a_form = fem.form(ufl.inner(normal_force(du), strain(u_)) * dx)

F0 = fem.Constant(domain, np.zeros((gdim,)))
L_form = fem.form(ufl.dot(F0, u_) * dx)

A = fem.petsc.assemble_matrix(a_form, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L_form)
with b.localForm() as loc:
    loc.set(0.0)

for component, value in enumerate(ROOF_LOAD_VECTOR):
    load_dofs = fem.locate_dofs_topological(
        V.sub(component), fdim, facets.find(ROOF_LOAD_TAG)
    )
    for dof in load_dofs:
        b.setValue(int(dof), float(value), addv=PETSc.InsertMode.ADD_VALUES)
b.assemble()

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
    u_grid.point_data["Deflection"] = u.x.array.reshape(-1, gdim)
    u_grid.set_active_vectors("Deflection")
    warped = u_grid.warp_by_vector("Deflection", factor=2000.0)
    warped.cell_data["Normal force"] = N.x.array

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain))
    plotter.add_mesh(grid, show_edges=True, line_width=4, color="royalblue", opacity=0.25)

    nmax = max(abs(N.x.array))
    plotter.add_mesh(
        warped,
        show_scalar_bar=True,
        scalars="Normal force",
        render_lines_as_tubes=True,
        style="wireframe",
        line_width=4,
        opacity=1,
        cmap="bwr",
        clim=[-nmax, nmax],
    )
    plotter.view_isometric()
    plotter.show()
    plotter.screenshot("stadium_truss_3D_after.png")


def values_in_original_node_order():
    dof_coords = u.function_space.tabulate_dof_coordinates()
    vals = u.x.array.reshape((-1, gdim))
    ordered = []
    for point in points:
        distances = np.linalg.norm(dof_coords - point, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > 1e-10:
            raise RuntimeError(f"Could not locate displacement at point {point}.")
        ordered.append(vals[idx])
    return np.array(ordered)


displacement_soln = values_in_original_node_order()

print("\n# Copy these arrays into a fe-jax 3D linearized Neumann test.")
print("points = np.array(")
print(repr(points.tolist()))
print(", dtype=np.float64)")
print("cells = np.array(")
print(repr(cells.tolist()))
print(", dtype=np.int64)")
print("support_nodes =", repr(support_nodes.tolist()))
print("roof_load_nodes =", repr(roof_load_nodes.tolist()))
print("roof_load_vector =", repr(ROOF_LOAD_VECTOR.tolist()))
print("displacement_soln = np.array(")
print(repr(displacement_soln.tolist()))
print(", dtype=np.float64)")

print("-" * 96)
print(f"{'node':^18}|{'coordinates':^35}|{'displacements':^35}")
print("-" * 96)
for name, x, v in zip(node_names, points, displacement_soln):
    coord_str = "[" + " ".join(f"{xi: 8.3f}" for xi in x) + "]"
    val_str = "[" + " ".join(f"{vi: .16e}" for vi in v) + "]"
    print(f"{name:>18} | {coord_str:>35} | {val_str:>35}")
