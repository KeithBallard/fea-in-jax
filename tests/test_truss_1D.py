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
    s = f"{val:.3e}"
    mantissa, exp = s.split('e')
    exp = int(exp)

    mantissa = f"{float(mantissa):.3f}".rstrip('0').rstrip('.')
    return f"{mantissa} \\cdot 10^{{{exp}}}"

def export_to_latex_table(points, u_truss, filename="table.txt"):
    with open(filename, "w") as f:
        f.write("\\begin{tabular}{c|c}\n")
        f.write("\\hline\n")

        f.write("\\textbf{point} & \\textbf{displacement} \\\\\n")
        f.write("\\hline\n")

        for xi, vi in zip(points, u_truss):
            x = float(xi[0])
            v = float(vi)

            coord = f"${f'{x:.3f}'.rstrip('0').rstrip('.')}$"
            val   = f"${sci_to_latex(v)}$"

            f.write(f"{coord} & {val} \\\\\n")

        f.write("\\end{tabular}\n")

n_elements = 4

def run_truss_1D_bar(n_elements, stretch_factor, label):
    points = np.linspace(0, 1, n_elements + 1, dtype=np.float32).reshape((-1, 1))
    soln   = points*(1+stretch_factor)-points
    cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.uint64)
    cell_domain_ids = np.zeros(cells.shape[0], dtype=np.int64)

    # Sizes of arrays
    U = 1  # number of solution components
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
    matrix_mat_params = jnp.array([1.0e9,1])  # E


    # Example using the isotropic constitutive relation
    element_batches_iso = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=1,
            connectivity_en=cells,
            constitutive_model=elastic_isotropic,
            material_params=matrix_mat_params,
        )
    ]
    # Set boundary conditions (first endpoint stays fixed, final endpoint goes from 1.0 -> 1.2
    bcs = ([DirichletBC(bc_type = BCType.NODE,component=0, index=0,value=0.0),
            DirichletBC(bc_type=BCType.NODE,component=0,index=n_elements,value=soln[-1,0])])
    u_iso, residual_iso, element_batches_iso = solve_bvp(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches_iso,
        boundary_conditions=bcs,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=1,
            linear_max_iter=5,
        ),
        plot_convergence=True,
    )
    plt.close()
    # print("\n*** Isotropic constitutive_model ***")
    # print("|R| = ", jnp.linalg.norm(residual_iso))
    # print(f"u = {u_iso}")
    # dirichlet_dofs = np.array([bc.index for bc in bcs])
    # dirichlet_values = np.array([bc.value for bc in bcs])
    # assert jnp.isclose(u_iso[dirichlet_dofs], dirichlet_values).all(), f"Dirichlet is not satisfied"

    # Example using the truss elements
    element_batches_truss = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=1,
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
            linear_max_iter=5,
        ),
        plot_convergence=True,
    )
    plt.savefig(get_output(f"solver_convergence_1D_{label}.png"))
    plt.close()

    export_to_latex_table(points,u_truss,filename = get_output(f"displacement_table_1D_{label}.txt"))
    print("\n*** Truss Elements! ***")
    print("|R| = ", jnp.linalg.norm(residual_truss))
    print('-'*24)
    print(f"{'initial':^9}|{'displacement':^14}")
    print('-'*24)
    for xi, vi in zip(points, u_truss):
        # Format each coordinate and value to 6 decimal places
        coord_str = f"{xi[0]:7.3f}"
        val_str = f"{vi: .3e}"
        print(f"{coord_str:^9}|{val_str:^14}")
    print("\n"*2)
    #dirichlet_dofs = np.array([bc.index for bc in bcs])
    #dirichlet_values = np.array([bc.value for bc in bcs])
    #assert jnp.isclose(u_truss[dirichlet_dofs], dirichlet_values).all(), f"Dirichlet is not satisfied"

    ##Check that the two solutions match! 
    #assert jnp.isclose(u_truss,u_iso).all(), "The solutions from the isotropic model and the truss model do NOT match!"
    #print("The solutions from the isotropic and truss models match (at least to JAX default precision)!")
    return u_truss, u_iso

@pytest.mark.parametrize("case_args",[
    (4,0.2,'1D_stretch'),
    (4,-0.2,'1D_compression')
])

def test_truss_1D_bar(case_args):
    u,ref_soln = run_truss_1D_bar(*case_args)
    assert jnp.isclose(u,ref_soln).all(), (
        f"Does not match isotropic solution: {case_args[-1]}! "
        f"Absolute error is {np.max(np.abs(u-ref_soln)):.3e}."
    )
