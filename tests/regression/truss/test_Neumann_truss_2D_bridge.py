from helper import *
import pytest
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)
import jax.extend
# from jax_smi import initialise_tracking
# initialise_tracking()

pytestmark = pytest.mark.truss

if __name__ == "__main__":
    print(jax.extend.backend.get_backend().platform)

def plot_truss(points,cells,linecolor='tab:blue',markercolor='k',marker=None,linestyle = None):
    for conn in cells:
        plt.plot(*points[conn].T,color=linecolor,linestyle = linestyle)
    plt.scatter(points[:,0],points[:,1],color=markercolor,marker=marker)

def run_truss_2D_bridge(
    label="2D_bridge",
    nonlinear_max_iter=30,
):
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
            [ 0.0000000000000000e+00,  0.0000000000000000e+00],
            [-4.0348069928191154e-06, -4.7413240142144382e-04],
            [ 1.9941712420270919e-06, -6.2699173349650900e-04],
            [-7.0585158235953546e-20, -6.6738402294532265e-04],
            [-1.9941712420272130e-06, -6.2699173349650911e-04],
            [ 4.0348069928190680e-06, -4.7413240142144393e-04],
            [ 0.0000000000000000e+00,  0.0000000000000000e+00],
            [-1.3716160425393349e-04, -4.5033755671635155e-04],
            [-8.8605854596248432e-05, -5.9051646249395884e-04],
            [-1.0327574913615736e-20, -6.3192317035040680e-04],
            [ 8.8605854596248378e-05, -5.9051646249395873e-04],
            [ 1.3716160425393341e-04, -4.5033755671635133e-04],
        ]
    )
    coordinate_soln = points + displacement_soln

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
    matrix_mat_params = jnp.array([200e3, 1], dtype=jnp.float64)  # E, A

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

    # -----------------------------------------------------------------------------
    #        coordinates        |                 displacements
    # -----------------------------------------------------------------------------
    # [    0.0     0.0     0.0] | [ 0.0000000000000000e+00,  0.0000000000000000e+00],
    # [    5.0     0.0     0.0] | [-4.0348069928191154e-06, -4.7413240142144382e-04],
    # [   10.0     0.0     0.0] | [ 1.9941712420270919e-06, -6.2699173349650900e-04],
    # [   15.0     0.0     0.0] | [-7.0585158235953546e-20, -6.6738402294532265e-04],
    # [   20.0     0.0     0.0] | [-1.9941712420272130e-06, -6.2699173349650911e-04],
    # [   25.0     0.0     0.0] | [ 4.0348069928190680e-06, -4.7413240142144393e-04],
    # [   30.0     0.0     0.0] | [ 0.0000000000000000e+00,  0.0000000000000000e+00],
    # [   25.0     3.3     0.0] | [-1.3716160425393349e-04, -4.5033755671635155e-04],
    # [   20.0     5.3     0.0] | [-8.8605854596248432e-05, -5.9051646249395884e-04],
    # [   15.0     6.0     0.0] | [-1.0327574913615736e-20, -6.3192317035040680e-04],
    # [   10.0     5.3     0.0] | [ 8.8605854596248378e-05, -5.9051646249395873e-04],
    # [    5.0     3.3     0.0] | [ 1.3716160425393341e-04, -4.5033755671635133e-04],
    bcs = (
        [
            DirichletBC(bc_type = BCType.NODE,component=0, index=0,value=displacement_soln[0][0]),
            DirichletBC(bc_type = BCType.NODE,component=1, index=0,value=displacement_soln[0][1]),
            DirichletBC(bc_type = BCType.NODE,component=0, index=6,value=displacement_soln[6][0]),
            DirichletBC(bc_type = BCType.NODE,component=1, index=6,value=displacement_soln[6][1]),
            NeumannBC(bc_type = BCType.NODE,component=1, index=1,value=-1),
            NeumannBC(bc_type = BCType.NODE,component=1, index=2,value=-1),
            NeumannBC(bc_type = BCType.NODE,component=1, index=3,value=-1),
            NeumannBC(bc_type = BCType.NODE,component=1, index=4,value=-1),
            NeumannBC(bc_type = BCType.NODE,component=1, index=5,value=-1),
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
        ),
        plot_convergence=False,
    )

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

    plt.figure(figsize=[12,3])
    plt.scatter(*points.T,label = 'initial')
    plot_truss(points+2000*u_truss,cells,linecolor='tab:orange',linestyle='solid',markercolor='tab:orange',marker='d')
    plt.scatter(*(points+2000*u_truss).T,marker='d',label = 'solution')
    plot_truss(points + 2000*displacement_soln,cells,linecolor='tab:green',linestyle='dashed',markercolor='tab:green',marker='x')
    plt.scatter(*(points + 2000*displacement_soln).T,marker='x',label = 'FEniCS')
    plt.legend()
    plt.subplots_adjust(left=0.03,right=0.99,top=0.98)
    plt.savefig(get_output(f"solution_{label}.png"))
    plt.close()

    # dirichlet_dofs = np.array([bc.index for bc in bcs])
    # dirichlet_values = np.array([bc.value for bc in bcs])
    # dirichlet_comp = np.array([bc.component for bc in bcs])
    # assert jnp.isclose(u_truss[dirichlet_dofs,dirichlet_comp], dirichlet_values).all(), f"Dirichlet is not satisfied"
    # print("Solution at least matches at the Dirichlet boundary conditions.\n")

    # assert jnp.isclose(u_truss,displacement_soln).all(), f"Does not match expected solution"
    # print("Solution matches the expected solution (copied form FEniCS)")

    return u_truss,displacement_soln

def test_truss_2D_bridge_linearized():
    u, ref_soln = run_truss_2D_bridge(
        label="2D_bridge_linearized",
        nonlinear_max_iter=1,
    )
    assert jnp.allclose(u, ref_soln, rtol=1e-11, atol=1e-12), (
        "The one-step linearized bridge solution does not match the "
        f"FEniCS small-displacement solution. Max absolute error is "
        f"{jnp.max(jnp.abs(u - ref_soln)):.3e}."
    )


@pytest.mark.skip(reason="Nonlinear reference solution still needs to be added.")
def test_truss_2D_bridge_nonlinear():
    run_truss_2D_bridge(
        label="2D_bridge_nonlinear",
        nonlinear_max_iter=30,
    )
