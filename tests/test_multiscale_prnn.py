from helper import *

from prnn import *

jax.config.update("jax_enable_x64", True)

# from jax_smi import initialise_tracking
# initialise_tracking()

settings = {
    "data_path": "datasets/gpCurves.data",
    "decoder_type": "SoftLayer",
    "input_norm": False,  # Note: keep false. Normalization has not yet been consistently implemented for computing losses.
    "output_norm": False,
    "mat_points": 2,  # Number of hidden layers related to material
    "feature_dim": 3,
    "output_dim": 3,
}

def load_params(filename):

    # Add .npy extension if not present
    if not filename.endswith(".npy"):
        filename = f"{filename}.npy"

    # Load the checkpoint
    checkpoint = np.load(filename, allow_pickle=True).item()

    print(f"best_val: {checkpoint['best_val']}")
    print(f"_epoch: {checkpoint['epoch']}")

    if checkpoint["best_params"] is not None:
        # Convert NumPy arrays back to JAX arrays
        best_params = jax.tree_util.tree_map(
            lambda x: jnp.array(x), checkpoint["best_params"]
        )
        return best_params
    else:
        params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint["params"])
        return params

model = PRNN(
    n_features=settings["feature_dim"],
    n_outputs=settings["output_dim"],
    n_matpts=settings["mat_points"],
    decoder_type=settings["decoder_type"],
)
print(
    f"New PRNN: Input (strain) size {model.n_features} - Material layer size (points) {model.n_matpts} - Output (stress) size {model.n_outputs}"
)
material = jax_j2.create_material()
params = load_params(
    os.path.join(os.path.dirname(__file__), "prnn/model_0_matpts_2_ncurve_16.npy")
)

@jax.jit
def elastic_prnn(eps_qdd: jnp.ndarray, internal_state_qi: jnp.ndarray):
    """
    A machine-learned constitive relation based on the PRNN architecture.

    Parameters
    ----------
    eps_qdd           : infinitesimal strain tensor, ndarray[float, (Q, D, D)]
    internal_state_qi : current internal state variables, ndarray[float, (Q, I)]

    Returns
    -------
    stress_qdd  : stress tensor, ndarray[float, (Q, D, D)]
    """
    assert eps_qdd.shape[2] == 2, "Only implemented for 2D strain"
    # Note: model, params, and material declared outside function
    eps_inputs = rank2_tensor_to_voigt(eps_qdd)
    stress_voigt_qs, new_internal_state_qi = model.apply(
        params, eps_inputs, internal_state_qi, material
    )
    stress_qdd = rank2_voigt_to_tensor(stress_voigt_qs)
    return stress_qdd, new_internal_state_qi


# Read in the mesh
mesh = meshio.read(get_mesh(f"rectangle.vtk"))
points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
cells = np.array([], dtype=np.uint64)
for cell_block in mesh.cells:
    if cell_block.type == "triangle":
        cells = np.array(cell_block.data, dtype=np.uint64)
print("# DoFs = ", 2 * points.shape[0])

# Sizes of arrays
U = 2  # number of solution components
V = points.shape[0]  # number of vertices
E = cells.shape[0]  # number of elements
F = V * U  # number of DoFs
fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=3,
)
Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points
I = 7 * model.n_matpts  # PRNN model requires 7 internal state variables

# Define random Dirichlet boundary conditions
min_xy = np.min(points, axis=0)
max_xy = np.max(points, axis=0)
left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]
# An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
dirichlet_bcs = []
dirichlet_values = []
# Fix left nodes along x-direction
dirichlet_bcs.append(np.zeros(shape=(left_points.shape[0], 2), dtype=np.uint64))
dirichlet_bcs[-1][:, 0] = left_points
dirichlet_bcs[-1][:, 1] = 0
dirichlet_values.append(np.zeros(shape=(dirichlet_bcs[-1].shape[0],)))
# Fix right nodes such that the model is subjected to 1% strain along x-axis
dirichlet_bcs.append(np.zeros(shape=(right_points.shape[0], 2), dtype=np.uint64))
dirichlet_bcs[-1][:, 0] = right_points
dirichlet_bcs[-1][:, 1] = 0
dirichlet_values.append(
    (max_xy[0] - min_xy[0]) / 1000.0 * mesh.points[right_points][:, 1]
)
print("Right side dirichlet_values = ", dirichlet_values[-1])
# Fix bottom nodes along y-direction
dirichlet_bcs.append(np.zeros(shape=(bottom_points.shape[0], 2), dtype=np.uint64))
dirichlet_bcs[-1][:, 0] = bottom_points
dirichlet_bcs[-1][:, 1] = 1
dirichlet_values.append(np.zeros(shape=(dirichlet_bcs[-1].shape[0],)))
# Convert the lists to combined arrays
dirichlet_bcs = np.vstack(dirichlet_bcs)
dirichlet_values = np.concat(dirichlet_values)

mat_params_eqm = jnp.zeros(shape=(cells.shape[0], Q, 2))
mat_params_eqm = mat_params_eqm.at[:, :, 0].set(3.45e9)  # E
mat_params_eqm = mat_params_eqm.at[:, :, 1].set(0.35)  # nu

element_batches = [
    ElementBatch(
        fe_type=fe_type,
        connectivity_en=cells,
        constitutive_model=elastic_prnn,
        material_params_eqm=mat_params_eqm,
        internal_state_eqi=jnp.zeros(shape=(cells.shape[0], Q, I)),
    ),
]

u_prev = jnp.zeros((V * U,))

for i, scale_factor in enumerate(np.linspace(0, 1, 10)):

    # Solve the boundary value problem
    u, residual, new_internal_state_beqi = solve_bvp(
        element_residual_func=linear_elasticity_residual,
        vertices_vd=points,
        element_batches=element_batches,
        u_0_g=u_prev,
        dirichlet_bcs=dirichlet_bcs,
        dirichlet_values=scale_factor * dirichlet_values,
        solver_options=SolverOptions(linear_solve_type=LinearSolverType.DIRECT_INVERSE_JNP),
    )
    print("|R| = ", jnp.linalg.norm(residual))

    u_prev = u

    # Write output
    mesh.point_data["u"] = u.reshape((points.shape[0], U))
    mesh.write(get_output(f"test_multiscale_prnn_out_{i}.vtk"))

os.system(
    f"zip {os.path.join(os.path.dirname(__file__), 'output', 'test_multiscale_prnn_out.zip')} {os.path.join(os.path.dirname(__file__), 'output', 'test_multiscale_prnn_out_*.vtk')}"
)
