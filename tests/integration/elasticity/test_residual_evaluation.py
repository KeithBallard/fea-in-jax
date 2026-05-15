import meshio

from fe_jax.helper import *

import numpy as np

# General notes:
# 1) It might be helpful to inherit from jax.array and add labels for axes to
#    help with debugging and enable a higher level description of operations.

def test_residual_evaluation_on_polygon_meshes():
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

        element_batches = [
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=U,
                connectivity_en=cells,
                constitutive_model=elastic_isotropic,
                material_params=mat_params_eqp,
            ),
        ]

        (
            ebc,
            assembly_map_b,
            constraint_system,
            jacobian_nnz,
            element_residual_func,
            f_ext,
        ) = preprocess_bvp(
            vertices_vd=points,
            element_batches=element_batches,
            element_residual_func=linear_elasticity_residual,
        )

        def residual_func(u_f):
            residual, _ = calculate_residual_w_constraints(
                element_residual_func=element_residual_func,
                ebc=ebc,
                assembly_map_b=assembly_map_b,
                u_f=u_f,
                constraints=constraint_system,
                f_ext=f_ext,
            )
            return residual

        R_f = timeit(
            f=residual_func,
            generated_kwargs={
                "u_f": lambda: jnp.array(np.random.rand(V * U))
            },
            time_jit=True,
            n_calls=2,
            timings_figure_filepath=f"timings/cpu_timing_{mesh_size}.png",
        )[0]
        print("R_f", R_f.shape)  # , R_f)
        assert R_f.shape == (V * U,)
