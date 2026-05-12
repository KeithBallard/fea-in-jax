from .vtms_structs import *
from ..fea import *
from ..contact import *
from ..linear_elasticity import *


@dataclass
class VTMSFiberMaterial:
    id: int
    E: float
    A: float


def solve_fiber_mechanics_bvp(
    fabric: VTMSFabric,
    materials: list[VTMSFiberMaterial],
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC],
    contact_search_radius: float,
):
    """
    TODO document
    """

    fe_type = FiniteElementType(
        cell_type=CellType.interval,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    n_dofs_per_basis = 3

    print(f"fabric.get_n_elements(): {fabric.get_n_elements()}")
    connectivity_en = np.zeros((fabric.get_n_elements(), 2), dtype=np.uint64)
    element_index = 0
    for i in range(fabric.fiber_offsets.shape[0] - 1):
        for j in range(fabric.fiber_offsets[i + 1] - fabric.fiber_offsets[i] - 1):
            connectivity_en[element_index] = [
                fabric.fiber_offsets[i] + j,
                fabric.fiber_offsets[i] + j + 1,
            ]
            element_index += 1

    material_params = np.zeros((connectivity_en.shape[0], 2), dtype=np.float64)
    element_index = 0
    for b_i in range(fabric.get_n_bundles()):
        mat = materials[fabric.get_material_id(b_i)]
        params = jnp.array([mat.E, mat.A])
        for f_i in range(fabric.get_n_fibers_in_bundle(b_i)):
            for e_i in range(fabric.get_fiber_n_elements(b_i, f_i)):
                material_params[element_index] = params
                element_index += 1

    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=n_dofs_per_basis,
            connectivity_en=connectivity_en,
            constitutive_model=elastic_truss,
            material_params=jnp.array(material_params),
        )
    ]

    point_fiber_ids = np.concatenate(
        [
            np.full((fabric.fiber_offsets[i + 1] - fabric.fiber_offsets[i],), i)
            for i in range(fabric.fiber_offsets.shape[0] - 1)
        ]
    )

    contact_fe_type = fe_type
    self_adjacency_block = 3
    contact_search_radius = contact_search_radius
    contact_params = jnp.array([10 * np.max(material_params[:,0]), 1, contact_search_radius])  # E_max, A, R

    def contact_pair_generator() -> list[ElementBatch]:
        contact_cells = contact_batch(
            points=fabric.points,
            point_fiber_ids=point_fiber_ids,
            adjacency_block=self_adjacency_block,
            radius=contact_search_radius,
        )
        return [
            ElementBatch(
                fe_type=contact_fe_type,
                n_dofs_per_basis=n_dofs_per_basis,
                connectivity_en=contact_cells,
                constitutive_model=elastic_contact_truss,
                material_params=contact_params,
            )
        ]

    u_truss, residual_truss, element_batches_truss = solve_bvp(
        element_residual_func=linear_truss_residual,
        vertices_vd=fabric.points,
        element_batches=element_batches,
        boundary_conditions=boundary_conditions,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=50,
            linear_max_iter=50,
        ),
        plot_convergence=False,
        contact_batch_generator=contact_pair_generator,
    )

    return u_truss
