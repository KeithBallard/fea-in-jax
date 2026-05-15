from fe_jax.helper import *
import pytest
import matplotlib.pyplot as plt
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
import jax.extend

pytestmark = pytest.mark.truss

if __name__ == '__main__':
    print(jax.extend.backend.get_backend().platform)

def run_truss_2element_Neumann():
    # points = np.array([[-5,0],[0,-8.66],[5,0]],dtype = jnp.float64)
    points = np.array([[-5,0],[0,-5*np.sqrt(3)],[5,0]],dtype = jnp.float64)
    cells = np.array([[0,1],[1,2]],dtype = jnp.int32)
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
    matrix_mat_params = jnp.array([10e6,0.1])

    # Set boundary conditions. Leave the (0,0) end point fixed, but take (2,1)->(2.4,1.2)
    # The displacement is in the direciton of the bar, so this should be the same as a 1D displacement.
    bcs = (
        [
            DirichletBC(bc_type=BCType.NODE, component=0, index=0, value=-0.0),
            DirichletBC(bc_type=BCType.NODE, component=1, index=0, value= 0.0),
            # DirichletBC(bc_type=BCType.NODE, component=0, index=1, value= 0.0),
            # DirichletBC(bc_type=BCType.NODE, component=1, index=1, value=-1732/150e3),
            DirichletBC(bc_type=BCType.NODE, component=0, index=2, value= 0.0),
            DirichletBC(bc_type=BCType.NODE, component=1, index=2, value= 0.0),
            NeumannBC(bc_type=BCType.NODE, component = 1, index=1, value = -1732),
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
            nonlinear_max_iter=1,
            linear_max_iter=30,
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

    # Check solution against Dirichlet boundary conditions
    # dirichlet_dofs = np.array([bc.index for bc in bcs if isinstance(bc,DirichletBC)])
    # dirichlet_values = np.array([bc.value for bc in bcs if isinstance(bc,DirichletBC)])
    # dirichlet_comp = np.array([bc.component for bc in bcs if isinstance(bc,DirichletBC)])
    # assert jnp.isclose(u_truss[dirichlet_dofs,dirichlet_comp], dirichlet_values).all(), f"Dirichlet is not satisfied"
    # print("Woo Hoo! Solution at least matches at the endopints\n")

    # plt.scatter(*points.T,label = 'initial')
    # plt.scatter(*(points+u_truss).T,marker='d',label = 'solution')
    # plt.legend()
    # plt.show()

    ref_soln = jnp.array([[0,0],[0,-1732/150e3],[0,0]])
    return u_truss, ref_soln

def test_truss_2element_Neumann():
    u, ref_soln = run_truss_2element_Neumann()
    assert jnp.allclose(u,ref_soln,rtol=1e-11,atol=1e-12)
