from helper import *
<<<<<<< Updated upstream
=======
import pytest
import time
import jetsci
>>>>>>> Stashed changes

import time
import numpy as np
import fe_jax.sparse_linear_solve as sls
from fe_jax.solve_cg import cg as cg_w_info

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.extend


def _block(x):
    return jax.block_until_ready(x)


def _time_call(label, repeats, fn):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = _block(fn())
        times.append(time.perf_counter() - start)
    arr = np.asarray(times)
    print(
        f"{label}: min={arr.min():.6f} s, "
        f"mean={arr.mean():.6f} s, last={arr[-1]:.6f} s"
    )
    return result, {"min": float(arr.min()), "mean": float(arr.mean()), "last": float(arr[-1])}

<<<<<<< Updated upstream

def _print_timing_summary(timings):
    print("\nTiming summary")
    print("--------------")
    print(f"{'path':38s} {'min (s)':>10s} {'mean (s)':>10s} {'last (s)':>10s}")
    for label, values in timings.items():
        print(
            f"{label:38s} "
            f"{values['min']:10.6f} "
            f"{values['mean']:10.6f} "
            f"{values['last']:10.6f}"
        )
    if "PETSc init+solve" in timings and "JAX assembled CG w/info" in timings:
        ratio = timings["PETSc init+solve"]["mean"] / timings["JAX assembled CG w/info"]["mean"]
        print(f"\nPETSc init+solve / JAX assembled CG mean ratio: {ratio:.3f}x")


def _diag_from_coo(A):
    diag_terms = jnp.where(A.row == A.col, A.data, 0.0)
    return jnp.zeros((A.shape[0],), dtype=A.data.dtype).at[A.row].add(diag_terms)


@jax.jit
def _jax_assembled_cg_with_info(A, rhs, diag):
    eps = 1e-12
    scale = jnp.maximum(jnp.max(jnp.abs(diag)), 1.0)
    safe_diag = jnp.maximum(jnp.abs(diag), eps * scale)
    preconditioner = lambda x: x / safe_diag
    x, info = cg_w_info(
        A=A,
        b=rhs,
        M=preconditioner,
        tol=1e-14,
        atol=1e-10,
        maxiter=100000,
    )
    return x, info["iterations"]


@jax.jit
def _jax_scipy_assembled_cg(A, rhs, diag):
    eps = 1e-12
    scale = jnp.maximum(jnp.max(jnp.abs(diag)), 1.0)
    safe_diag = jnp.maximum(jnp.abs(diag), eps * scale)
    preconditioner = lambda x: x / safe_diag
    x, _ = jax.scipy.sparse.linalg.cg(
        A=A,
        b=rhs,
        M=preconditioner,
        tol=1e-14,
        atol=1e-10,
        maxiter=100000,
    )
    return x


def _petsc_init_solve(A, rhs, petsc_solver_type, petsc_precond):
    ctx = sls.__petsc_init(A.shape, A.data, A.row, A.col, petsc_solver_type, petsc_precond)
    return sls.__petsc_solve(ctx, rhs)


def _residual_norm(A, x, rhs):
    return float(jnp.linalg.norm(A @ x - rhs))


def _build_problem(mesh_file):
    mesh = meshio.read(get_mesh(mesh_file))
=======
    # Read in the mesh
    mesh = meshio.read(get_mesh("microscale_2D_r2.vtk"))
>>>>>>> Stashed changes
    points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
    cells = np.array(mesh.cells[0].data, dtype=np.uint64)
    mesh.cell_data["DomainIDs"][0] = np.array(mesh.cell_data["DomainIDs"][0], dtype=np.int64)
    cell_domain_ids = mesh.cell_data["DomainIDs"][0].flatten()
    print("# DoFs = ", 2 * points.shape[0])

    U = 2
    V = points.shape[0]
    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=2,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=3,
    )
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]

    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
    right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
    bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
    top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]

    dirichlet_bcs, dirichlet_values = build_dirichlet_arrays_from_lists(
        point_indices=[left_points, right_points, bottom_points, top_points],
        components=[0, 0, 1, 1],
        values=[0.0, (max_xy[0] - min_xy[0]) / 100.0, 0.0, 0.0],
    )

    matrix_cells = cells[cell_domain_ids == 0]
    fiber_cells = cells[cell_domain_ids == 1]

    @jax.jit
    def get_properties():
        matrix_mat_params_eqm = jnp.zeros(shape=(matrix_cells.shape[0], Q, 2))
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(3.45e9)
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 1].set(0.35)

        fiber_mat_params_eqm = jnp.zeros(shape=(fiber_cells.shape[0], Q, 4))
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(26e9)
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(26e9)
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(0.7218543046357615)
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(7.55e9)
        return matrix_mat_params_eqm, fiber_mat_params_eqm

    matrix_mat_params_eqm, fiber_mat_params_eqm = get_properties()
    print(fiber_mat_params_eqm[0, 0, :])

    def make_element_batches():
        return [
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=2,
                connectivity_en=matrix_cells,
                constitutive_model=elastic_isotropic,
                material_params=matrix_mat_params_eqm,
            ),
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=2,
                connectivity_en=fiber_cells,
                constitutive_model=elastic_orthotropic,
                material_params=fiber_mat_params_eqm,
            ),
        ]

    return mesh, points, element_batches, jnp.zeros(shape=(V * U)), dirichlet_bcs, dirichlet_values


<<<<<<< Updated upstream

def _solve_bvp_residual(
    points,
    element_batches,
    u0,
    dirichlet_bcs,
    dirichlet_values,
    solver_type,
    petsc_solver_type,
    preconditioner,
):
    _, residual, _ = solve_bvp(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        u_0_g=u0,
        dirichlet_bcs=dirichlet_bcs,
        dirichlet_values=dirichlet_values,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType[solver_type],
            linear_precond_type=PreconditionerType[preconditioner],
            petsc_solve_type=petsc_solver_type,
        ),
        plot_convergence=False,
    )
    return residual


def _run_compare(mesh_file, petsc_solver_type, preconditioner, repeats):
    _, points, element_batches, u0, dirichlet_bcs, dirichlet_values = _build_problem(mesh_file)

    print("\nCOMPARE mode: timing full solve_bvp paths.")
    print("This avoids the standalone assembleRHS wrapper on this branch, which is not wired the same way as solve_bvp.")

    print("\nWarming full solve_bvp paths.")
    _block(
        _solve_bvp_residual(
            points,
            element_batches,
            u0,
            dirichlet_bcs,
            dirichlet_values,
            "CG_JAX_SCIPY_W_INFO",
            petsc_solver_type,
            preconditioner,
        )
    )
    _block(
        _solve_bvp_residual(
            points,
            element_batches,
            u0,
            dirichlet_bcs,
            dirichlet_values,
            "PETSC",
            petsc_solver_type,
            preconditioner,
        )
    )

    timings = {}
    print("\nTiming full solve_bvp paths.")
    _, timings["full solve_bvp JAX w/info"] = _time_call(
        "full solve_bvp JAX w/info",
        repeats,
        lambda: _solve_bvp_residual(
            points,
            element_batches,
            u0,
            dirichlet_bcs,
            dirichlet_values,
            "CG_JAX_SCIPY_W_INFO",
            petsc_solver_type,
            preconditioner,
        ),
    )
    _, timings["full solve_bvp PETSc"] = _time_call(
        "full solve_bvp PETSc",
        repeats,
        lambda: _solve_bvp_residual(
            points,
            element_batches,
            u0,
            dirichlet_bcs,
            dirichlet_values,
            "PETSC",
            petsc_solver_type,
            preconditioner,
        ),
    )
    _print_timing_summary(timings)
=======
    def block_bvp_result(result):
        u, residual, element_batches = result[:3]
        u.block_until_ready()
        residual.block_until_ready()
        return u, residual, element_batches

    def time_solve(label, solve_func, n_calls=3):
        start = time.perf_counter()
        result = block_bvp_result(solve_func())
        first_call_time = time.perf_counter() - start

        times = []
        for _ in range(n_calls):
            start = time.perf_counter()
            result = block_bvp_result(solve_func())
            times.append(time.perf_counter() - start)

        print(
            f"{label}: first call {first_call_time:.6f} sec, "
            f"avg {sum(times) / len(times):.6f} sec "
            f"(n={n_calls}, min={min(times):.6f}, max={max(times):.6f})"
        )
        return result, first_call_time, times

    nonlinear_relative_tol = 1e-5
    nonlinear_absolute_tol = 1e-14
    linear_relative_tol = 1e-5
    linear_absolute_tol = 1e-14
    linear_max_iter = 5000
    petsc_diagnostics = False
    petsc_options = [
        jetsci.SolverOptions(
            nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
            linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
            linear_solve_type=jetsci.PETScLinearSolverType.CG,
            nonlinear_relative_tol=nonlinear_relative_tol,
            nonlinear_absolute_tol=nonlinear_absolute_tol,
            linear_max_iter=linear_max_iter,
            linear_relative_tol=linear_relative_tol,
            linear_absolute_tol=linear_absolute_tol,
        )
    ]

    def run_petsc():
        result = solve_bvp_PETSc(
            vertices_vd=points,
            element_batches=make_element_batches(),
            element_residual_func=linear_elasticity_residual,
            boundary_conditions=bcs,
            multipoint_constraints=None,
            u_0_g=u_0,
            diagnostics=petsc_diagnostics,
            petsc_solver_options=petsc_options[0],
            destroy_solver=False,
            return_petsc_solver_options=True,
        )
        petsc_options[0] = result[3]
        return result[:3]

    def run_jax():
        return solve_bvp(
            element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=make_element_batches(),
            u_0_g=u_0,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                linear_precond_type=PreconditionerType.JACOBI,
                nonlinear_relative_tol=nonlinear_relative_tol,
                nonlinear_absolute_tol=nonlinear_absolute_tol,
                linear_max_iter=linear_max_iter,
                linear_relative_tol=linear_relative_tol,
                linear_absolute_tol=linear_absolute_tol,
            ),
        )

    try:
        petsc_result, _, _ = time_solve("PETSc solve_bvp_PETSc", run_petsc, n_calls=3)
    finally:
        solver_key = petsc_options[0].solver_key
        if solver_key is not None:
            jetsci.petsc_snes.differentiable_snes.unregister_primitive_context(solver_key)
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)
    jax_result, _, _ = time_solve("JAX solve_bvp", run_jax, n_calls=3)

    u_petsc, residual_petsc, _ = petsc_result
    u, residual, element_batches = jax_result

    residual_petsc_norm = jnp.linalg.norm(residual_petsc)
    residual_jax_norm = jnp.linalg.norm(residual)
    print("|R| PETSc = ", residual_petsc_norm)
    print("|R| JAX   = ", residual_jax_norm)
    print("|u_petsc - u_jax| = ", jnp.linalg.norm(u_petsc - u))

    diff_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
        linear_solve_type=jetsci.PETScLinearSolverType.CG,
        nonlinear_relative_tol=nonlinear_relative_tol,
        nonlinear_absolute_tol=nonlinear_absolute_tol,
        linear_max_iter=linear_max_iter,
        linear_relative_tol=linear_relative_tol,
        linear_absolute_tol=linear_absolute_tol,
    )
    solve_phi, phi_0, x_0, _, diff_options = build_differentiable_bvp_PETSc_solve(
        vertices_vd=points,
        element_batches=make_element_batches(),
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=bcs,
        multipoint_constraints=None,
        u_0_g=u_0,
        diagnostics=False,
        petsc_solver_options=diff_options,
    )

    try:
        response_weights = jnp.linspace(
            0.5,
            1.5,
            x_0.shape[0],
            dtype=x_0.dtype,
        )
        response_weights = response_weights / jnp.linalg.norm(response_weights)
        phi_direction = jnp.sin(
            jnp.arange(phi_0.shape[0], dtype=phi_0.dtype) + 1.0
        )
        phi_dot = 1e-3 * phi_0 * phi_direction

        def scalar_response(active_phi):
            u_active = solve_phi(active_phi, x_0)
            return jnp.vdot(response_weights, u_active)

        def make_element_batches_from_phi(active_phi):
            matrix_size = matrix_mat_params_eqm.size
            matrix_phi = active_phi[:matrix_size].reshape(matrix_mat_params_eqm.shape)
            fiber_phi = active_phi[matrix_size:].reshape(fiber_mat_params_eqm.shape)
            return [
                ElementBatch(
                    fe_type=fe_type,
                    n_dofs_per_basis=2,
                    connectivity_en=matrix_cells,
                    constitutive_model=elastic_isotropic,
                    material_params=matrix_phi,
                ),
                ElementBatch(
                    fe_type=fe_type,
                    n_dofs_per_basis=2,
                    connectivity_en=fiber_cells,
                    constitutive_model=elastic_orthotropic,
                    material_params=fiber_phi,
                ),
            ]

        def jax_scalar_response(active_phi):
            u_active, _, _ = solve_bvp(
                element_residual_func=linear_elasticity_residual,
                vertices_vd=points,
                element_batches=make_element_batches_from_phi(active_phi),
                u_0_g=u_0,
                boundary_conditions=bcs,
                solver_options=SolverOptions(
                    linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                    linear_precond_type=PreconditionerType.JACOBI,
                    nonlinear_relative_tol=nonlinear_relative_tol,
                    nonlinear_absolute_tol=nonlinear_absolute_tol,
                    linear_max_iter=linear_max_iter,
                    linear_relative_tol=linear_relative_tol,
                    linear_absolute_tol=linear_absolute_tol,
                ),
            )
            return jnp.vdot(response_weights, u_active)

        response, response_dot = jax.jvp(
            scalar_response,
            (phi_0,),
            (phi_dot,),
        )
        response.block_until_ready()
        response_dot.block_until_ready()

        eps = 1.0
        response_plus = scalar_response(phi_0 + eps * phi_dot)
        response_minus = scalar_response(phi_0 - eps * phi_dot)
        finite_difference_dot = (response_plus - response_minus) / (2.0 * eps)
        finite_difference_dot.block_until_ready()

        jax_response, jax_response_dot = jax.jvp(
            jax_scalar_response,
            (phi_0,),
            (phi_dot,),
        )
        jax_response.block_until_ready()
        jax_response_dot.block_until_ready()

        derivative_abs_error = jnp.abs(response_dot - finite_difference_dot)
        derivative_rel_error = derivative_abs_error / (
            jnp.abs(finite_difference_dot) + 1e-14
        )
        finite_difference_signal = jnp.abs(response_plus - response_minus)
        expected_finite_difference_signal = jnp.abs(2.0 * eps * response_dot)
        petsc_jax_abs_error = jnp.abs(response_dot - jax_response_dot)
        petsc_jax_rel_error = petsc_jax_abs_error / (
            jnp.abs(jax_response_dot) + 1e-14
        )
        print("PETSc d response / d phi dot =", response_dot)
        print("JAX d response / d phi dot =", jax_response_dot)
        print("Finite-difference d response / d phi dot =", finite_difference_dot)
        print("Finite-difference response signal =", finite_difference_signal)
        print("Expected finite-difference response signal =", expected_finite_difference_signal)
        print("PETSc vs finite-difference abs error =", derivative_abs_error)
        print("PETSc vs finite-difference rel error =", derivative_rel_error)
        print("PETSc vs JAX abs error =", petsc_jax_abs_error)
        print("PETSc vs JAX rel error =", petsc_jax_rel_error)
        assert (float(petsc_jax_abs_error) < 1e-10) or (
            float(petsc_jax_rel_error) < 5e-2
        )
        if float(expected_finite_difference_signal) > 1e-10:
            assert (float(derivative_abs_error) < 1e-10) or (
                float(derivative_rel_error) < 5e-2
            )
        else:
            print(
                "Skipping finite-difference assertion because the expected response "
                "difference is below the solve-noise threshold."
            )
    finally:
        solver_key = diff_options.solver_key
        if solver_key is not None:
            jetsci.petsc_snes.differentiable_snes.unregister_primitive_context(solver_key)
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)
>>>>>>> Stashed changes

    ratio = timings["full solve_bvp PETSc"]["mean"] / timings["full solve_bvp JAX w/info"]["mean"]
    print(f"\nFull solve_bvp PETSc / JAX mean ratio: {ratio:.3f}x")

def _run_original(mesh_file, solver_type, petsc_solver_type, preconditioner):
    mesh, points, element_batches, u0, dirichlet_bcs, dirichlet_values = _build_problem(mesh_file)
    U = 2

    start = time.time()
    u, residual, element_batches = solve_bvp(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        u_0_g=u0,
        dirichlet_bcs=dirichlet_bcs,
        dirichlet_values=dirichlet_values,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType[solver_type],
            linear_precond_type=PreconditionerType[preconditioner],
            petsc_solve_type=petsc_solver_type,
        ),
        plot_convergence=False,
    )
    jax.block_until_ready(residual)
    print("full took", time.time() - start)

    dirichlet_dofs = U * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1]
    assert jnp.isclose(u[dirichlet_dofs], dirichlet_values).all()


<<<<<<< Updated upstream
meshFile = sys.argv[1]
solverType = sys.argv[2]
petscSolverType = int(sys.argv[3])
preconditioner = sys.argv[4]
repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 3

print(petscSolverType)

if solverType == "COMPARE":
    _run_compare(meshFile, petscSolverType, preconditioner, repeats)
else:
    _run_original(meshFile, solverType, petscSolverType, preconditioner)



=======
test_microscale_bvp()
>>>>>>> Stashed changes
