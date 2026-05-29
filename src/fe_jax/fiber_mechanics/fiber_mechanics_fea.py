from .vtms_structs import *
from .vtk_exporter import *
from ..postprocess import get_output
from ..fea import *
from ..contact import *
from ..linear_elasticity import *
from typing import List
from copy import deepcopy

@dataclass
class VTMSFiberMaterial:
    id: int
    E: float
    A: float


def solve_fiber_mechanics_bvp(
    fabric: VTMSFabric,
    materials: list[VTMSFiberMaterial],
    boundary_conditions: list[DirichletBC | NeumannBC | PeriodicBC] | list[list[DirichletBC | NeumannBC| PeriodicBC]],
    contact_search_radius: float,
    solver_options: SolverOptions,
    pseudotime_iters: int = 1,
    plot_convergence: bool = False,
    blow_up_threshold: float = jnp.inf,
    filename_base: str | None = None,
    # boundary_conditions_per_step: list[
    #     list[DirichletBC | NeumannBC | PeriodicBC]
    # ]
    # | None = None,
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
    contact_params = jnp.array([10 * np.max(material_params[:,0]), np.max(material_params[:,1]), contact_search_radius])  # E_max, A, R

    def contact_pair_generator() -> list[ElementBatch] | None:
        contact_cells = contact_batch(
            points=fabric.points,
            point_fiber_ids=point_fiber_ids,
            adjacency_block=self_adjacency_block,
            radius=contact_search_radius,
        )
        if contact_cells.shape[0] == 0: return []
        return [
            ElementBatch(
                fe_type=contact_fe_type,
                n_dofs_per_basis=n_dofs_per_basis,
                connectivity_en=contact_cells,
                constitutive_model=elastic_contact_truss,
                material_params=contact_params,
            )
        ]

    if filename_base is not None:
        write_vtk(fabric,get_output(filename=f"{filename_base}_0.vtk", subdir="contact"))

    # Normalize boundary_conditions to a per-step schedule.
    # Static input: [bc1, bc2, ...] -> [[bc1, bc2, ...], ..., [bc1, bc2, ...]]
    # Dynamic input: [[...], [...], ...] stays as-is.
    if len(boundary_conditions) == 0:
        boundary_conditions = [[] for _ in range(pseudotime_iters)]
    elif isinstance(boundary_conditions[0], (DirichletBC, NeumannBC, PeriodicBC)):
        boundary_conditions = [
            list(boundary_conditions) for _ in range(pseudotime_iters)
        ]
    else:
        if len(boundary_conditions) != pseudotime_iters:
            raise ValueError(
                "If boundary_conditions is already a per-step schedule, it must have "
                f"exactly pseudotime_iters={pseudotime_iters} entries."
            )

    # if boundary_conditions_per_step is not None:
    #     if len(boundary_conditions_per_step) != pseudotime_iters:
    #         raise ValueError(
    #             "boundary_conditions_per_step must contain exactly "
    #             f"pseudotime_iters={pseudotime_iters} entries, but got "
    #             f"{len(boundary_conditions_per_step)}."
    #         )

    for i in range(pseudotime_iters):
        print(f"\n \n   pseudo-timestep i = {i+1}\n \n")
        # bcs_i = (
        #     boundary_conditions_per_step[i]
        #     if boundary_conditions_per_step is not None
        #     else boundary_conditions
        # )
        u_truss, residual_truss, element_batches_truss = solve_bvp(
            element_residual_func=linear_truss_residual,
            vertices_vd=fabric.points,
            u_0_g=None if i==0 else u_truss,
            element_batches=element_batches,
            boundary_conditions=boundary_conditions[i],
            solver_options=solver_options,
            plot_convergence=plot_convergence,
            contact_batch_generator=contact_pair_generator,
        )
        # u_truss = u_truss.reshape((-1,fabric.points.shape[1]))
        print(f"\nmax(||u||) = {np.linalg.norm(u_truss.reshape((-1,fabric.points.shape[1])),axis=1).max()}\n")
        # fabric.points = fabric.points + np.array(u_truss)

        if filename_base is not None:
            temp_fab = deepcopy(fabric)
            temp_fab.points = temp_fab.points + np.array(u_truss.reshape((-1,temp_fab.points.shape[1])))
            write_vtk(temp_fab,get_output(filename=f"{filename_base}_{i+1}.vtk", subdir="contact"))
        if jnp.isnan(u_truss).any() or jnp.isinf(u_truss).any() or np.linalg.norm(u_truss.reshape((-1,fabric.points.shape[1])),axis=1).max()>blow_up_threshold:
            raise RuntimeError(f"Nonlinear solve diverged: displacement magnitude exceeded threshold ({blow_up_threshold})")

    return u_truss, residual_truss, element_batches_truss
