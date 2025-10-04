import meshio

from helper import *

import numpy as np

# General notes:
# 1) It might be helpful to inherit from jax.array and add labels for axes to
#    help with debugging and enable a higher level description of operations.

for mesh_size in [0.05, 0.01, 0.005, 0.001]:

    mesh = meshio.read(get_mesh(f"polygon_mesh_{mesh_size}.vtk"))

    points = np.array(mesh.points, dtype=np.float64)[:, 0:2]
    cells = np.array(mesh.cells[1].data, dtype=np.uint64)

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

    tmp_mat_params = np.random.rand(E, Q, 2)
    tmp_mat_params[..., 0] = 90e9 * tmp_mat_params[..., 0] + 10e9
    mat_params_eqp = jnp.array(tmp_mat_params)

    # Setup element batches
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            connectivity_en=cells,
            constitutive_model=elastic_isotropic,
            material_params_eqm=mat_params_eqp,
            internal_state_eqi=jnp.zeros(shape=(cells.shape[0], Q, 0)),
        ),
    ]

    # Setup inputs
    x_bend = [
        mesh_to_jax(vertices=points, cells=b.connectivity_en)
        for b in element_batches
    ]
    xi_bqp, W_bq = zip(*[get_quadrature(fe_type=b.fe_type) for b in element_batches])
    W_bq = list(W_bq)
    phi_bqn, dphi_dxi_bqnp = zip(
        *[
            eval_basis_and_derivatives(fe_type=b.fe_type, xi_qp=xi_bqp[i])
            for i, b in enumerate(element_batches)
        ]
    )
    dphi_dxi_bqnp = list(dphi_dxi_bqnp)

    material_params_beqm = [b.material_params_eqm for b in element_batches]
    internal_state_beqi = [b.internal_state_eqi for b in element_batches]

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    print("x_bend[0]", x_bend[0].shape)  # , x_n)
    print("xi_bqp[0]", xi_bqp[0].shape)  # , xi_qp)
    print("phi_bqn[0]", phi_bqn[0].shape)  # , phi_qp)
    print("dphi_dxi_bqnp[0]", dphi_dxi_bqnp[0].shape)  # , dphi_dxi_qp)
    print("mat_params_eqp", mat_params_eqp.shape)  # , mat_params_qp)

    dirichlet_values_g = jnp.zeros((V * U, ))
    dirichlet_mask_g = jnp.zeros((V * U, ))

    residual_isv_func_w_dirichlet = lambda u_g: calculate_residual_w_dirichlet(
        element_residual_func=jax.tree_util.Partial(linear_elasticity_residual),
        constitutive_model_b=[jax.tree_util.Partial(elastic_isotropic)],
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_g,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
    )
    
    R_e = timeit(
        f=residual_isv_func_w_dirichlet,
        fixed_kwargs={},
        generated_kwargs={
            "u_g": lambda: jnp.array(np.random.rand(V * U))
        },
        time_jit=True,
        n_calls=20,
        timings_figure_filepath=f'timings/cpu_timing_{mesh_size}.png'
    )[0][0]
    print("R_e", R_e.shape)  # , R_e)
