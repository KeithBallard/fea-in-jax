from fe_jax.helper import *
import pytest
import matplotlib.pyplot as plt
import jax.extend
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.truss

if __name__ == "__main__":
    print(jax.extend.backend.get_backend().platform)

# from jax_smi import initialise_tracking
# initialise_tracking()

def run_truss_2D_bar(n_elements: int, end_point: tuple,stretch_factor: tuple,label: str):
    """
    This test case assums ALL rods start at (0,0,0),
    this makes the stretch/compression easy to apply.
    """
    xn,yn = end_point
    stretch  = 1+np.array(stretch_factor)
    points=np.array(
        (
            np.linspace(0,xn,n_elements + 1),
            np.linspace(0,yn,n_elements + 1),
        )
    ).T
    soln = points*stretch - points
    cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.uint64)
    # points = np.linspace(0, 1, n_elements + 1, dtype=np.float32).reshape((-1, 1))
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
    matrix_mat_params = jnp.array([1.0e9,1.0])  # E

    # Set boundary conditions. Leave the (0,0) end point fixed, but take (xn,yn)->(sx*xn,sy*yn)
    # The displacement is in the direciton of the bar, so this should be the same as a 1D displacement.
    bcs = (
        [
            DirichletBC(bc_type = BCType.NODE, component = 0, index = 0         , value = 0.0),
            DirichletBC(bc_type = BCType.NODE, component = 1, index = 0         , value = 0.0),
            DirichletBC(bc_type = BCType.NODE, component = 0, index = n_elements, value = soln[-1,0]),
            DirichletBC(bc_type = BCType.NODE, component = 1, index = n_elements, value = soln[-1,1]),
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
            nonlinear_max_iter=5,
            linear_max_iter=20,
            # nonlinear_relative_tol=1e-18,
            # nonlinear_absolute_tol=1e-18,
        ),
        plot_convergence=True,
    )

    plt.savefig(get_output(f"solver_convergence_{label}.png"))
    plt.close()
    u_truss = u_truss.reshape((-1,2))
    print("\n*** Truss Elements! ***")
    print("|R| = ", jnp.linalg.norm(residual_truss))
    print('-'*45)
    print(f"{'initial':^19}|{'displacement':^24}")
    print('-'*45)
    for x, v in zip(points, u_truss):
        # Format each coordinate and value to 6 decimal places
        coord_str = "[" + " ".join(f"{xi:7.3f}" for xi in x) + "]"
        val_str = "[" + " ".join(f"{vi: .3e}" for vi in v) + "]"
        print(f"{coord_str:>18} | {val_str:>24}")
    print("\n"*2)

    # Check solution against Dirichlet boundary conditions
    dirichlet_dofs = np.array([bc.index for bc in bcs])
    dirichlet_values = np.array([bc.value for bc in bcs])
    dirichlet_comp = np.array([bc.component for bc in bcs])

    plt.scatter(*points.T,label = 'initial')
    plt.scatter(*(points+u_truss).T,marker='d',label = 'solution')
    plt.scatter(*(points+soln).T,marker='x',label = 'truth')
    plt.legend()
    plt.savefig(get_output(f"solution_{label}.png"))
    plt.close()

    return u_truss,soln
    # assert jnp.isclose(u_truss[dirichlet_dofs,dirichlet_comp].reshape((-1,2)), dirichlet_values.reshape((-1,2))).all(), f"Dirichlet is not satisfied"
    # print("Woo Hoo! Solution at least matches at the endopints\n")

@pytest.mark.parametrize("case_args",[
    (6,(2,1),(0.2,0.2),"2D_stretch"),
    (6,(2,1),(-0.2,-0.2),"2D_compression"),
    (6,(2,1),(0.2,0),"2D_skew")
])

def test_truss_2D_bar(case_args):
    u, ref_soln = run_truss_2D_bar(*case_args)
    assert jnp.isclose(u,ref_soln).all(), (
        f"does not match expected solution: {case_args[-1]}! "
        f"Absolute error is {np.max(np.abs(u-ref_soln)):.3e}."
    )
