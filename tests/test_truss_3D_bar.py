from helper import *
import pytest
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import jax.extend

if __name__ == "__main__":
    print(jax.extend.backend.get_backend().platform)
# from jax_smi import initialise_tracking
# initialise_tracking()

def sci_to_latex(val):
    """
    Convert a float to LaTeX scientific notation: a \\times 10^{b}
    """
    s = f"{val:.3e}"           # e.g., -1.234e-05
    mantissa, exp = s.split('e')
    exp = int(exp)             # remove leading zeros like e-05 -> -5

    # Clean mantissa (remove trailing zeros if desired)
    mantissa = f"{float(mantissa):.3f}".rstrip('0').rstrip('.')

    return f"{mantissa} \\cdot 10^{{{exp}}}"


def export_to_latex_table(points, u_truss, filename="table.txt"):
    with open(filename, "w") as f:
        f.write("\\begin{tabular}{ccc|ccc}\n")
        f.write("\\hline\n")

        f.write("\\multicolumn{3}{c|}{\\textbf{points}} & "
                "\\multicolumn{3}{c}{\\textbf{displacement}} \\\\\n")

        f.write("\\hline\n")

        for p, v in zip(points, u_truss):
            # coord_vals = [f"${xi:0.3g}$" for xi in p]
            coord_vals = [f"${f'{xi:.3f}'.rstrip('0').rstrip('.')}$" for xi in p]
            val_vals   = [f"${sci_to_latex(vi)}$" for vi in v]

            row = " & ".join(coord_vals + val_vals)
            f.write(row + " \\\\\n")

        f.write("\\end{tabular}\n")

def run_truss_3D_bar(n_elements: int, end_point: tuple, stretch_factors: tuple, label: str):
    """
    This test case assums ALL rods start at (0,0,0),
    this makes the stretch/compression easy to apply.
    """
    xn,yn,zn = end_point
    stretch = 1+np.array(stretch_factors)
    points=np.vstack((
        np.linspace(0,xn,n_elements+1),
        np.linspace(0,yn,n_elements+1),
        np.linspace(0,zn,n_elements+1),
    )).T
    soln = points*stretch - points
    cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.uint64)
    # points = np.linspace(0, 1, n_elements + 1, dtype=np.float32).reshape((-1, 1))
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
    matrix_mat_params = jnp.array([1e9])  # E

    # Set boundary conditions. Leave the (0,0) end point fixed, but take (2,1)->(2.4,1.2)
    # The displacement is in the direciton of the bar, so this should be the same as a 1D displacement.
    bcs = (
        [
            DirichletBC(bc_type=BCType.NODE, component=0, index=0         , value=0.0),
            DirichletBC(bc_type=BCType.NODE, component=1, index=0         , value=0.0),
            DirichletBC(bc_type=BCType.NODE, component=2, index=0         , value=0.0),
            DirichletBC(bc_type=BCType.NODE, component=0, index=n_elements, value=soln[-1,0]),
            DirichletBC(bc_type=BCType.NODE, component=1, index=n_elements, value=soln[-1,1]),
            DirichletBC(bc_type=BCType.NODE, component=2, index=n_elements, value=soln[-1,2])
        ]
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
            nonlinear_max_iter=30,
            linear_max_iter=20,
        ),
        plot_convergence=True,
    )
    plt.savefig(get_output(f"solver_convergence_{label}.png"))
    plt.close()

    u_truss = u_truss.reshape((-1,3))
    export_to_latex_table(points,u_truss,filename = get_output(f"displacement_table_{label}.txt"))
    print("\n*** Truss Elements! ***")
    print(f"*** {label:^15} ***")
    print("|R| = ", jnp.linalg.norm(residual_truss))
    print('-'*64)
    print(f"{'initial':^27}|{'final':^35}")
    print('-'*64)
    for p, v in zip(points, u_truss):
        # Format each coordinate and value to 6 decimal places
        coord_str = "[" + " ".join(f"{xi:7.3f}" for xi in p) + "]"
        val_str = "[" + " ".join(f"{vi: .3e}" for vi in v) + "]"
        print(f"{coord_str:>26} | {val_str:>35}")
    print("\n"*2)

    # Check solution against Dirichlet boundary conditions
    # dirichlet_dofs = np.array([bc.index for bc in bcs])
    # dirichlet_values = np.array([bc.value for bc in bcs])
    # dirichlet_comp = np.array([bc.component for bc in bcs])
    # assert jnp.isclose(u_truss[dirichlet_dofs,dirichlet_comp], dirichlet_values).all(), f"Dirichlet is not satisfied"
    # print("Woo Hoo! Solution at least matches at the endopints\n")

    # Check solutions against "known" solution. 
    return u_truss, soln
    # assert jnp.isclose(points+u_truss,np.vstack((x_soln,y_soln,z_soln)).T).all(), "does not match expected solution!"
    # print("Solution matches expected results!")

@pytest.mark.parametrize("case_args",[
    (6,(2,1,3),(0.2,0.2,0.2),"3D_stretch"),
    (6,(2,1,3),(-0.2,-0.2,-0.2),"3D_compression"),
    (4,(2,1,3),(0.2,0.1,0),"3D_skew_stretch")
])

def test_truss_3D_bar(case_args) -> None:
    u, ref_soln = run_truss_3D_bar(*case_args)
    assert jnp.isclose(u,ref_soln).all(), f"does not match expected solution: {case_args[-1]}!"

