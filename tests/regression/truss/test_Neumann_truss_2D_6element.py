from helper import *
import pytest
import matplotlib.pyplot as plt
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
import jax.extend

pytestmark = pytest.mark.truss

if __name__ == '__main__':
    print(jax.extend.backend.get_backend().platform)

def plot_truss(points,cells,linecolor='tab:blue',markercolor='k',marker=None,linestyle = None):
    for conn in cells:
        plt.plot(*points[conn].T,color=linecolor,linestyle = linestyle)
    plt.scatter(points[:,0],points[:,1],color=markercolor,marker=marker)

def run_truss_6element_Neumann(
    ll=100,
    P=1000,
    label="2D_6element_Neumann",
    nonlinear_max_iter=30,
):
    # points = np.array([[-5,0],[0,-8.66],[5,0]],dtype = jnp.float64)
    points = np.array([[0,ll],[ll,ll],[2*ll,ll],[0,0],[ll,0]],dtype = jnp.float64)

    cells = np.array([[0,1],[1,2],[1,3],[1,4],[2,4],[3,4]],dtype = jnp.int32)
    cell_domain_ids = np.zeros(cells.shape[0], dtype=np.int32)

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
    matrix_mat_params = jnp.array([30e6,0.5],dtype=jnp.float64)
    E = matrix_mat_params[0]
    A = matrix_mat_params[1]
    k = A*E/ll
    ref_soln = jnp.array(
        [
            [0,0],
            [2*P/k,-(2+np.sqrt(2))*np.sqrt(2)*P/k],
            [3*P/k,-(7+4*np.sqrt(2))*P/k],
            [0,0],
            [-P/k, -(3+2*np.sqrt(2))*P/k]
        ],
        dtype=jnp.float64
    )

    # Set boundary conditions. Leave the (0,0) end point fixed, but take (2,1)->(2.4,1.2)
    # The displacement is in the direciton of the bar, so this should be the same as a 1D displacement.
    bcs = (
        [
            DirichletBC(bc_type=BCType.NODE, component=0, index=0, value=-0.0),
            DirichletBC(bc_type=BCType.NODE, component=1, index=0, value= 0.0),
            # DirichletBC(bc_type=BCType.NODE, component=0, index=1, value= 0.0),
            # DirichletBC(bc_type=BCType.NODE, component=1, index=1, value=-1732/150e3),
            DirichletBC(bc_type=BCType.NODE, component=0, index=3, value= 0.0),
            DirichletBC(bc_type=BCType.NODE, component=1, index=3, value= 0.0),
            NeumannBC(bc_type=BCType.NODE, component = 1, index=2, value = -P),
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
            nonlinear_max_iter=nonlinear_max_iter,
            linear_max_iter=30,
            nonlinear_relative_tol = 1e-13,
            nonlinear_absolute_tol = 1e-10,
        ),
        plot_convergence=False,
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

    plt.figure(figsize=[6,3])
    explode_factor = 200
    plot_truss(points+explode_factor*u_truss,cells,linecolor='tab:blue',linestyle='solid',markercolor='tab:blue',marker='.')
    plt.scatter(*points.T,label = 'initial')
    plot_truss(points+explode_factor*u_truss,cells,linecolor='tab:orange',linestyle='solid',markercolor='tab:orange',marker='d')
    plt.scatter(*(points+explode_factor*u_truss).T,marker='d',label = 'solution')
    plot_truss(points + explode_factor*ref_soln,cells,linecolor='tab:green',linestyle='dashed',markercolor='tab:green',marker='x')
    plt.scatter(*(points + explode_factor*ref_soln).T,marker='x',label = 'analytic')
    plt.legend()
    # plt.subplots_adjust(left=0.03,right=0.99,top=0.98)
    plt.axis('equal')
    plt.savefig(get_output(f"solution_{label}.png"))

    return u_truss, ref_soln

def test_truss_6element_Neumann_linearized():
    u, ref_soln = run_truss_6element_Neumann(
        label="2D_6element_Neumann_linearized",
        nonlinear_max_iter=1,
    )
    assert jnp.allclose(u, ref_soln, rtol=1e-11, atol=1e-12), (
        "The one-step linearized truss solution does not match the "
        f"analytic small-displacement solution. Max absolute error is "
        f"{jnp.max(jnp.abs(u - ref_soln)):.3e}."
    )


@pytest.mark.skip(reason="Nonlinear reference solution still needs to be added.")
def test_truss_6element_Neumann_nonlinear():
    run_truss_6element_Neumann(
        label="2D_6element_Neumann_nonlinear",
        nonlinear_max_iter=30,
    )
