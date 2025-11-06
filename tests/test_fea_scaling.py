from helper import *

max_n_subdivisions = 3
n_subsequent_calls = 3
case_name = "batched_jit"
n_batches = 1

cpu_n_cols = math.ceil(math.sqrt(max_n_subdivisions + 1))
cpu_n_rows = math.ceil((max_n_subdivisions + 1) / cpu_n_cols)
cpu_fig, cpu_axes = plt.subplots(cpu_n_rows, cpu_n_cols, figsize=(12, 8))
cpu_axes = tuple(chain.from_iterable(cpu_axes))  # unravel the tuple of tuples
with poll_cpu() as cpu_poll:

    n_dofs = []
    solve_times = []
    n_dofs_2 = []
    first_call_times = []
    solve_memory: dict = {}
    for n_subdivisions in range(max_n_subdivisions + 1):

        cpu_poll.restart()

        # Read in the mesh
        mesh = meshio.read(get_mesh('polygon_mesh_0.5.vtk'))
        cpu_poll.mark_event("Finished reading mesh")

        # Refine the mesh
        v, t = refine_tri_mesh(
            vertices=np.array(mesh.points, dtype=np.float64),
            cells=np.array(mesh.cells[1].data, dtype=np.int32),
            number_of_subdivisions=n_subdivisions,
        )
        points = np.array(v[:, 0:2], dtype=np.float32)
        cells = np.array(t, dtype=np.int64)

        # Sizes of arrays
        U = 2  # number of solution components
        V = points.shape[0]  # number of vertices
        E = cells.shape[0]  # number of elements
        M = 2  # number of material parameters
        F = V * U  # number of DoFs
        fe_type = FiniteElementType(
            cell_type=CellType.triangle,
            family=ElementFamily.P,
            basis_degree=1,
            lagrange_variant=LagrangeVariant.equispaced,
            quadrature_type=QuadratureType.default,
            quadrature_degree=2,
        )
        Q = get_quadrature(fe_type=fe_type)[0].shape[0] # number of quadrature points
        

        print(f"Beginning subdivision {n_subdivisions} w/ # of DoFs {F}")

        # Define random Dirichlet boundary conditions
        boundary_points = find_tri_mesh_boundary_verts(cells=cells)
        # An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
        # Constrain very boundary point to have a random displacement
        dirichlet_bcs = np.zeros((U * boundary_points.shape[0], 2), dtype=np.uint64)
        for i in range(boundary_points.shape[0]):
            for j in range(U):
                dirichlet_bcs[U * i + j, 0] = i
                dirichlet_bcs[U * i + j, 1] = j
        # print(f'dirichlet_bcs: {dirichlet_bcs}')
        # Values of the Dirichlet boundary conditions matching 'dirichlet_bcs'
        dirichlet_values = 0.001 * np.random.rand(dirichlet_bcs.shape[0])

        # Set material properties at the quadrature point level randomly seeded such that
        # E = [90e9, 100e9] and nu = 0.25
        tmp_mat_params = np.random.rand(E, Q, M)
        tmp_mat_params[..., 0] = 90e9 * tmp_mat_params[..., 0] + 10e9
        tmp_mat_params[..., 1] = 0.25
        mat_params_eqm = jnp.array(tmp_mat_params)

        element_batches = [
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=U,
                connectivity_en=batch_cells,
                constitutive_model=elastic_isotropic,
                material_params_eqm=batch_mat_params_eqm
            )
            for batch_cells, batch_mat_params_eqm in zip(
                np.array_split(cells, n_batches, axis=0),
                np.array_split(mat_params_eqm, n_batches, axis=0),
            )
        ]

        cpu_poll.mark_event("Finished setup")

        # Solve the boundary value problem
        result, times, jit_time, first_call_time, peak_memory = timeit(
            f=solve_bvp,
            fixed_kwargs={
                "element_residual_func": linear_elasticity_residual,
                "vertices_vd": points,
                "element_batches": element_batches,
                "u_0_g": jnp.zeros(shape=(V * U)),
                "dirichlet_bcs": dirichlet_bcs,
                "dirichlet_values": dirichlet_values,
                "solver_options": SolverOptions(linear_solve_type=LinearSolverType.CG_JAXOPT),
                "profile_memory": True,
            },
            generated_kwargs={},
            time_jit=True,
            n_calls=n_subsequent_calls,
            timings_figure_filepath="",
            return_timing=True,
            return_memory=True,
        )

        cpu_poll.mark_event(f"Finished {n_subsequent_calls} solve_bvp calls")
        cpu_poll.stop()
        cpu_poll.get_plt_fig(cpu_axes[n_subdivisions], legend=False)
        cpu_axes[n_subdivisions].set_title(f"DoFs: {F}")

        n_dofs.extend([F] * len(times))
        solve_times.extend(times)
        n_dofs_2.extend([F])
        first_call_times.extend([first_call_time])

        # memory stats are specific to each device
        for device, d in peak_memory.items():
            if device not in solve_memory.keys():
                solve_memory[device] = []
            solve_memory[device].extend([d["bytes"]])

# plots
backend = jax.default_backend()
case_name = backend if len(case_name) == 0 else f"{case_name}_{backend}"

cpu_axes[-1].legend()  # only create the legend for the bottom right subplot
cpu_fig.tight_layout()
cpu_fig.savefig(get_output("test_fea_scaling_cpu_profile.png"), dpi=300)

# timings plot
np.save(get_output(f"test_fea_scaling_{case_name}_first_call_x"), n_dofs_2)
np.save(get_output(f"test_fea_scaling_{case_name}_first_call_y"), first_call_times)
np.save(get_output(f"test_fea_scaling_{case_name}_subseq_call_x"), n_dofs)
np.save(get_output(f"test_fea_scaling_{case_name}_subseq_call_y"), solve_times)
fig, ax = plt.subplots()
ax.scatter(n_dofs_2, first_call_times, label="First Call (JIT)")
ax.scatter(n_dofs, solve_times, label=f"Subsequent Calls ({n_subsequent_calls})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("# DoFs")
ax.set_ylabel("Time to Solve Linear Elastic BVP (s)")
ax.set_title(f"Scaling for {case_name}")
ax.legend()
fig.savefig(get_output(f"test_fea_scaling_{case_name}.png"), dpi=300)

plt.clf()
for device, data in solve_memory.items():
    np.save(
        get_output(f"test_fea_scaling_{case_name}_{device.replace(':','_')}_peak_memory_y"),
        data,
    )
    plt.scatter(n_dofs_2, data, label=f"{device}")

# Function to format the Y-axis labels
from matplotlib.ticker import FuncFormatter


def format_bytes(x, pos):
    x = int(x)
    for i, label in enumerate(["b", "kb", "Mb", "Gb", "Tb"]):
        if x < 1000 ** (i + 1):
            for y in [100, 10, 1]:
                if x / (1000**i) >= y:
                    break
            return f"{y} {label}"


plt.xscale("log")
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_bytes))
plt.xlabel("# DoFs")
plt.ylabel("Peak Memory Usage (bytes)")
plt.title(f"Scaling for {case_name}")
plt.legend()
plt.savefig(get_output(f"test_fea_scaling_{case_name}_memory.png"), dpi=300)
