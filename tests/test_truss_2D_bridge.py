from helper import *

jax.config.update("jax_enable_x64", True)

import jax.extend

print(jax.extend.backend.get_backend().platform)
# from jax_smi import initialise_tracking
# initialise_tracking()

def plot_truss(points,cells,linecolor='tab:blue',markercolor='k',marker=None,linestyle = None):
    for conn in cells:
        plt.plot(*points[conn].T,color=linecolor,linestyle = linestyle)
    plt.scatter(points[:,0],points[:,1],color=markercolor,marker=marker)

l = 5.0
h1 = 3.33
h2 = 5.33
h3 = 6.0
points=np.array(
    [[i*l,0.0] for i in range(7)] +
    [
        [5.0*l, h1 ],
        [4.0*l, h2 ],
        [3.0*l, h3 ],
        [2.0*l, h2 ],
        [1.0*l, h1 ]
    ]
)

cells = np.array(
    [[i,(i+1)%points.shape[0]] for i in range(points.shape[0])] + # Exterior members
    [[b,t] for b,t in zip(range(1,6),range(11,6,-1))] + # Vertical members
    [[b,t] for b,t in zip(range(1,5),range(10,6,-1))] + # up-rightwards diagonal members
    [[b,t] for b,t in zip(range(2,6),range(11,7,-1))],  # up-leftwards diagonal members
    dtype=np.int64
)
displacement_soln = np.array(
    [
        [ 0.000e+00,  0.000e+00],
        [-4.035e-06, -4.741e-04],
        [ 1.994e-06, -6.270e-04],
        [-7.059e-20, -6.674e-04],
        [-1.994e-06, -6.270e-04],
        [ 4.035e-06, -4.741e-04],
        [ 0.000e+00,  0.000e+00],
        [-1.372e-04, -4.503e-04],
        [-8.861e-05, -5.905e-04],
        [-1.033e-20, -6.319e-04],
        [ 8.861e-05, -5.905e-04],
        [ 1.372e-04, -4.503e-04]
    ]
)
coordinate_soln = points + displacement_soln*2000

# points = np.linspace(0, 1, n_elements + 1, dtype=np.float32).reshape((-1, 1))
# cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.uint64)
cell_domain_ids = np.zeros(cells.shape[0], dtype=np.int64)

# Sizes of arrays
U = 2  # number of solution components
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
matrix_mat_params = jnp.array([200e3])  # E

# Set boundary conditions. Taken from running FEniCS example and manually copying over.
# This is pretty crude, but should work for this purpose.
# I point to the coordinates that will get boundary conditions
# ---------------------------------------------------
#        coordinates        |     displacements
# ---------------------------------------------------
# [  0.000   0.000   0.000] | [ 0.000e+00  0.000e+00] <-- 0
# [  5.000   0.000   0.000] | [-4.035e-06 -4.741e-04] <-- 1
# [ 10.000   0.000   0.000] | [ 1.994e-06 -6.270e-04] <-- 2
# [ 15.000   0.000   0.000] | [-7.059e-20 -6.674e-04] <-- 3
# [ 20.000   0.000   0.000] | [-1.994e-06 -6.270e-04] <-- 4
# [ 25.000   0.000   0.000] | [ 4.035e-06 -4.741e-04] <-- 5
# [ 30.000   0.000   0.000] | [ 0.000e+00  0.000e+00] <-- 6
# [ 25.000   3.330   0.000] | [-1.372e-04 -4.503e-04]
# [ 20.000   5.330   0.000] | [-8.861e-05 -5.905e-04]
# [ 15.000   6.000   0.000] | [-1.033e-20 -6.319e-04]
# [ 10.000   5.330   0.000] | [ 8.861e-05 -5.905e-04]
# [  5.000   3.330   0.000] | [ 1.372e-04 -4.503e-04]
bcs = (
    [
        DirichletBC(bc_type = BCType.NODE,component=0, index=0,value=coordinate_soln[0][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=0,value=coordinate_soln[0][1]),
        DirichletBC(bc_type = BCType.NODE,component=0, index=6,value=coordinate_soln[6][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=6,value=coordinate_soln[6][1]),
        DirichletBC(bc_type = BCType.NODE,component=0, index=1,value=coordinate_soln[1][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=1,value=coordinate_soln[1][1]),
        DirichletBC(bc_type = BCType.NODE,component=0, index=2,value=coordinate_soln[2][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=2,value=coordinate_soln[2][1]),
        DirichletBC(bc_type = BCType.NODE,component=0, index=3,value=coordinate_soln[3][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=3,value=coordinate_soln[3][1]),
        DirichletBC(bc_type = BCType.NODE,component=0, index=4,value=coordinate_soln[4][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=4,value=coordinate_soln[4][1]),
        DirichletBC(bc_type = BCType.NODE,component=0, index=5,value=coordinate_soln[5][0]),
        DirichletBC(bc_type = BCType.NODE,component=1, index=5,value=coordinate_soln[5][1]),
    ]
)

# Example using the truss elements
element_batches_truss = [
    ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=2,
        connectivity_en=cells,
        constitutive_model=elastic_truss,
        material_params=matrix_mat_params,
    )
]

u_truss, residual_truss, element_batches_truss = solve_bvp(
    element_residual_func=linear_truss_residual,
    vertices_vd=points,
    element_batches=element_batches_truss,
    boundary_conditions=bcs,
    solver_options=SolverOptions(
        linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        # linear_precond_type=PreconditionerType.JACOBI,
        # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
        nonlinear_max_iter=10,
        linear_max_iter=10,
    ),
    plot_convergence=True,
)

u_truss = u_truss.reshape((-1,2))
print("\n*** Truss Elements! ***")
print("|R| = ", jnp.linalg.norm(residual_truss))
print('-'*45)
print(f"{'initial':^19}|{'final':^24}")
print('-'*45)
for x, v in zip(points, u_truss):
    # Format each coordinate and value to 6 decimal places
    coord_str = "[" + " ".join(f"{xi:7.3f}" for xi in x) + "]"
    val_str = "[" + " ".join(f"{vi: .3e}" for vi in v) + "]"
    print(f"{coord_str:>18} | {val_str:>24}")
print("\n"*2)

plt.figure(figsize=[12,3])
plt.scatter(*points.T,label = 'initial')
plot_truss(u_truss,cells,linecolor='tab:orange',linestyle='dashed',markercolor='tab:orange',marker='d')
plt.scatter(*u_truss.T,marker='d',label = 'solution')
plot_truss(coordinate_soln,cells,linecolor='tab:green',linestyle='dashed',markercolor='tab:green',marker='x')
plt.scatter(*coordinate_soln.T,marker='x',label = 'truth')
plt.legend()
plt.subplots_adjust(left=0.03,right=0.99,top=0.98)
plt.show()

dirichlet_dofs = np.array([bc.index for bc in bcs])
dirichlet_values = np.array([bc.value for bc in bcs])
dirichlet_comp = np.array([bc.component for bc in bcs])
assert jnp.isclose(u_truss[dirichlet_dofs,dirichlet_comp], dirichlet_values).all(), f"Dirichlet is not satisfied"
print("Solution at least matches at the Dirichlet boundary conditions.\n")
