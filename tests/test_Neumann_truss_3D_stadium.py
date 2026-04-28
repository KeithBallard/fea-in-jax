from helper import *
import pytest
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import jax.extend
# from jax_smi import initialise_tracking
# initialise_tracking()

if __name__ == "__main__":
    print(jax.extend.backend.get_backend().platform)

def plot_truss(points,cells,linecolor='tab:blue',markercolor='k',marker=None,linestyle = None):
    for conn in cells:
        plt.plot(*points[conn].T,color=linecolor,linestyle = linestyle)
    plt.scatter(points[:,0],points[:,1],color=markercolor,marker=marker)

def run_truss_3d_stadium(
    label="3D_stadium",
    nonlinear_max_iter=30,
):
    points = np.array(
        [
            [0.0, -5.0, 0.0],
            [6.0, -5.0, 0.0],
            [12.0, -5.0, 0.0],
            [18.0, -5.0, 0.0],
            [24.0, -5.0, 0.0],
            [30.0, -5.0, 0.0],
            [0.0, 5.0, 0.0],
            [6.0, 5.0, 0.0],
            [12.0, 5.0, 0.0],
            [18.0, 5.0, 0.0],
            [24.0, 5.0, 0.0],
            [30.0, 5.0, 0.0],
            [0.0, -7.0, 3.0],
            [6.0, -7.0, 5.0],
            [12.0, -7.0, 6.2],
            [18.0, -7.0, 6.2],
            [24.0, -7.0, 5.0],
            [30.0, -7.0, 3.0],
            [0.0, 7.0, 3.0],
            [6.0, 7.0, 5.0],
            [12.0, 7.0, 6.2],
            [18.0, 7.0, 6.2],
            [24.0, 7.0, 5.0],
            [30.0, 7.0, 3.0]
        ], dtype=np.float64
    )
    cells = np.array(
        [
            [0, 1], [6, 7], [12, 13], [18, 19], [0, 13], [12, 1], [6, 19], [18, 7], [12, 19], [18, 13], [0, 7], [6, 1], [1, 2], [7, 8], [13, 14], [19, 20], [1, 14], [13, 2], [7, 20], [19, 8], [13, 20], [19, 14], [1, 8], [7, 2], [2, 3], [8, 9], [14, 15], [20, 21], [2, 15], [14, 3], [8, 21], [20, 9], [14, 21], [20, 15], [2, 9], [8, 3], [3, 4], [9, 10], [15, 16], [21, 22], [3, 16], [15, 4], [9, 22], [21, 10], [15, 22], [21, 16], [3, 10], [9, 4], [4, 5], [10, 11], [16, 17], [22, 23], [4, 17], [16, 5], [10, 23], [22, 11], [16, 23], [22, 17], [4, 11], [10, 5], [0, 6], [12, 18], [0, 12], [6, 18], [1, 7], [13, 19], [1, 13], [7, 19], [2, 8], [14, 20], [2, 14], [8, 20], [3, 9], [15, 21], [3, 15], [9, 21], [4, 10], [16, 22], [4, 16], [10, 22], [5, 11], [17, 23], [5, 17], [11, 23]
        ], dtype=np.int64
    )
    support_nodes = [0, 6, 5, 11]
    roof_load_nodes = [13, 14, 15, 16, 19, 20, 21, 22]
    roof_load_vector = [0.05, 0.15, -2.0]
    displacement_soln = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.00012924281952014744, -0.00010651408216167996, 0.00015634702249776029],
            [-4.960831586183486e-05, -0.0001612037055316308, -0.00011613551456631368],
            [5.51988244410856e-05, -0.00016136774437881823, -0.00011552678861935293],
            [0.00013280701693267326, -0.00010721961778808632, 0.00015717225331715742],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.705064771627908e-05, -8.961781184322092e-05, -0.0011541897005777314],
            [1.405731519432167e-05, -0.00017757983313410362, -0.0013439904770942515],
            [-8.4668066150635e-06, -0.0001774157942869174, -0.001343381751147291],
            [-3.348645030374754e-05, -8.891227621681668e-05, -0.001153364469758333],
            [0.0, 0.0, 0.0],
            [-0.000618853589531296, 0.014452656907186586, 0.009495629220266703],
            [5.737722161801169e-05, 0.01655878038062992, 0.00673592850144245],
            [9.2068656470442e-05, 0.017701508452614385, 0.005646655895200835],
            [-8.469291147387069e-05, 0.01770148168566411, 0.00564763154119877],
            [-5.0502181791857526e-05, 0.016558706565545107, 0.00673800446076033],
            [0.0006241914162381837, 0.014452489862853576, 0.009495100159279886],
            [0.0008165592000013639, 0.014469740824177459, -0.009582479114563886],
            [0.000319932542223105, 0.016622224977382874, -0.007741003975901025],
            [7.369397142908133e-05, 0.017748434878844945, -0.007093878576042033],
            [-6.631822643249675e-05, 0.01774846164579519, -0.007092902930044091],
            [-0.0003130575023969379, 0.016622298792467613, -0.007738928016583114],
            [-0.0008112213732944624, 0.0144699078685104, -0.009583008175550661]
        ], dtype=np.float64
    )

    # points=np.array(
    #     [[i*l,0.0] for i in range(7)] +
    #     [
    #         [5.0*l, h1 ],
    #         [4.0*l, h2 ],
    #         [3.0*l, h3 ],
    #         [2.0*l, h2 ],
    #         [1.0*l, h1 ]
    #     ]
    # )

    # cells = np.array(
    #     [[i,(i+1)%points.shape[0]] for i in range(points.shape[0])] + # Exterior members
    #     [[b,t] for b,t in zip(range(1,6),range(11,6,-1))] + # Vertical members
    #     [[b,t] for b,t in zip(range(1,5),range(10,6,-1))] + # up-rightwards diagonal members
    #     [[b,t] for b,t in zip(range(2,6),range(11,7,-1))],  # up-leftwards diagonal members
    #     dtype=np.int64
    # )
    # displacement_soln = np.array(
    #     [
    #         [ 0.0000000000000000e+00,  0.0000000000000000e+00],
    #         [-4.0348069928191154e-06, -4.7413240142144382e-04],
    #         [ 1.9941712420270919e-06, -6.2699173349650900e-04],
    #         [-7.0585158235953546e-20, -6.6738402294532265e-04],
    #         [-1.9941712420272130e-06, -6.2699173349650911e-04],
    #         [ 4.0348069928190680e-06, -4.7413240142144393e-04],
    #         [ 0.0000000000000000e+00,  0.0000000000000000e+00],
    #         [-1.3716160425393349e-04, -4.5033755671635155e-04],
    #         [-8.8605854596248432e-05, -5.9051646249395884e-04],
    #         [-1.0327574913615736e-20, -6.3192317035040680e-04],
    #         [ 8.8605854596248378e-05, -5.9051646249395873e-04],
    #         [ 1.3716160425393341e-04, -4.5033755671635133e-04],
    #     ]
    # )

    # points = np.linspace(0, 1, n_elements + 1, dtype=np.float32).reshape((-1, 1))
    # cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.uint64)
    cell_domain_ids = np.zeros(cells.shape[0], dtype=np.int64)

    # Sizes of arrays
    U = 3  # number of solution components
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

    support_nodes = [0, 6, 5, 11]
    roof_load_nodes = [13, 14, 15, 16, 19, 20, 21, 22]
    roof_load_vector = [0.05, 0.15, -2.0]
    bcs = (
        [DirichletBC(bc_type = BCType.NODE,component=c, index=node,value= 0.) for node in support_nodes for c in range(3)] +
        [NeumannBC(bc_type = BCType.NODE,component=c, index=node,value=v) for node in roof_load_nodes for c,v in enumerate(roof_load_vector)]
    )

    # Example using the truss elements
    element_batches_truss = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=3,
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
            linear_max_iter=100,
        ),
        plot_convergence=False,
    )

    u_truss = u_truss.reshape((-1,3))
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

    # plt.figure(figsize=[12,3])
    # plt.scatter(*points.T,label = 'initial')
    # plot_truss(points+100*u_truss,cells,linecolor='tab:orange',linestyle='solid',markercolor='tab:orange',marker='d')
    # plt.scatter(*(points+100*u_truss).T,marker='d',label = 'solution')
    # plot_truss(points + 100*displacement_soln,cells,linecolor='tab:green',linestyle='dashed',markercolor='tab:green',marker='x')
    # plt.scatter(*(points + 100*displacement_soln).T,marker='x',label = 'FEniCS')
    # plt.legend()
    # plt.subplots_adjust(left=0.03,right=0.99,top=0.98)
    # plt.savefig(get_output(f"solution_{label}.png"))
    # plt.close()

    # dirichlet_dofs = np.array([bc.index for bc in bcs])
    # dirichlet_values = np.array([bc.value for bc in bcs])
    # dirichlet_comp = np.array([bc.component for bc in bcs])
    # assert jnp.isclose(u_truss[dirichlet_dofs,dirichlet_comp], dirichlet_values).all(), f"Dirichlet is not satisfied"
    # print("Solution at least matches at the Dirichlet boundary conditions.\n")

    # assert jnp.isclose(u_truss,displacement_soln).all(), f"Does not match expected solution"
    # print("Solution matches the expected solution (copied form FEniCS)")

    return u_truss, displacement_soln

def test_truss_3D_stadium_linearized():
    u, ref_soln = run_truss_3d_stadium(
        label="3D_stadium_linearized",
        nonlinear_max_iter=1,
    )
    assert jnp.allclose(u, ref_soln, rtol=1e-10, atol=1e-9), (
        "The one-step linearized 3D stadium truss solution does not match the "
        f"FEniCS small-displacement solution. Max absolute error is "
        f"{jnp.max(jnp.abs(u - ref_soln)):.3e}."
    )


@pytest.mark.skip(reason="Nonlinear reference solution still needs to be added.")
def test_truss_3D_stadium_nonlinear():
    run_truss_3d_stadium(
        label="3D_stadium_nonlinear",
        nonlinear_max_iter=30,
    )
