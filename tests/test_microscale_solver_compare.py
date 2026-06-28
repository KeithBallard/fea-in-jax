from helper import *

import time
import numpy as np

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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


def _print_summary(timings):
    print("\nTiming summary")
    print("--------------")
    print(f"{'path':36s} {'min (s)':>10s} {'mean (s)':>10s} {'last (s)':>10s}")
    for label, values in timings.items():
        print(
            f"{label:36s} "
            f"{values['min']:10.6f} "
            f"{values['mean']:10.6f} "
            f"{values['last']:10.6f}"
        )

    if "PETSc assembled" in timings and "JAX assembled CG" in timings:
        ratio = timings["PETSc assembled"]["mean"] / timings["JAX assembled CG"]["mean"]
        print(f"\nPETSc assembled / JAX assembled CG mean ratio: {ratio:.3f}x")


def _build_problem(mesh_name="microscale_2D_r0.vtk"):
    mesh = meshio.read(get_mesh(mesh_name))
    points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
    cells = np.array(mesh.cells[0].data, dtype=np.uint64)
    mesh.cell_data["DomainIDs"][0] = np.array(mesh.cell_data["DomainIDs"][0], dtype=np.int64)
    cell_domain_ids = mesh.cell_data["DomainIDs"][0].flatten()

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

    element_batches = [
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

    return points, element_batches, jnp.zeros(shape=(V * U)), dirichlet_bcs, dirichlet_values


@jax.jit
def _jax_assembled_cg(A, b, diag):
    safe_diag = jnp.maximum(jnp.abs(diag), 1e-12 * jnp.maximum(jnp.max(jnp.abs(diag)), 1.0))
    preconditioner = lambda x: x / safe_diag
    x, _ = jax.scipy.sparse.linalg.cg(
        A=A,
        b=b,
        M=preconditioner,
        tol=1e-14,
        atol=1e-10,
    )
    return x


def _residual_norm(A, x, b):
    return float(jnp.linalg.norm(A @ x - b))


def main():
    mesh_name = sys.argv[1] if len(sys.argv) > 1 else "microscale_2D_r0.vtk"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    petsc_solver_type = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    points, element_batches, u0, dirichlet_bcs, dirichlet_values = _build_problem(mesh_name)
    print("mesh:", mesh_name)
    print("# DoFs =", u0.shape[0])
    print("timing repeats:", repeats)
    print("petsc solver type:", petsc_solver_type)

    print("\nAssembling Jacobian/RHS once for solve-only comparison.")
    start = time.perf_counter()
    J = assembleJacobian(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        u_0_g=u0,
        dirichlet_bcs=dirichlet_bcs,
        dirichlet_values=dirichlet_values,
    )
    _block(J.data)
    jac_time = time.perf_counter() - start

    start = time.perf_counter()
    rhs = assembleRHS(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        u_0_g=u0,
        dirichlet_bcs=dirichlet_bcs,
        dirichlet_values=dirichlet_values,
    )
    rhs = -rhs
    _block(rhs)
    rhs_time = time.perf_counter() - start

    diag = jnp.zeros((u0.shape[0],), dtype=J.data.dtype).at[J.row].add(jnp.where(J.row == J.col, J.data, 0.0))
    print("nnz:", J.nse)
    print(f"Jacobian construction: {jac_time:.6f} s")
    print(f"RHS construction: {rhs_time:.6f} s")

    print("\nWarming solve paths.")
    _block(_jax_assembled_cg(J, rhs, diag))
    ctx = __petsc_init(J.shape, J.data, J.row, J.col, petsc_solver_type, 0)
    _block(__petsc_solve(ctx, rhs))

    print("\nTiming solve-only paths.")
    timings = {}
    x_jax, timings["JAX assembled CG"] = _time_call(
        "JAX assembled CG",
        repeats,
        lambda: _jax_assembled_cg(J, rhs, diag),
    )
    x_petsc, timings["PETSc assembled"] = _time_call(
        "PETSc assembled",
        repeats,
        lambda: __petsc_solve(ctx, rhs),
    )

    print("\nResiduals")
    print("JAX assembled CG ||Ax-b||:", _residual_norm(J, x_jax, rhs))
    print("PETSc assembled ||Ax-b||:", _residual_norm(J, x_petsc, rhs))
    print("||x_petsc - x_jax||:", float(jnp.linalg.norm(x_petsc - x_jax)))

    _print_summary(timings)
    print("\nConstruction times")
    print(f"Jacobian construction: {jac_time:.6f} s")
    print(f"RHS construction: {rhs_time:.6f} s")
    print(f"JAX construction + mean solve: {jac_time + rhs_time + timings['JAX assembled CG']['mean']:.6f} s")
    print(f"PETSc construction + mean solve: {jac_time + rhs_time + timings['PETSc assembled']['mean']:.6f} s")


if __name__ == "__main__":
    main()
