from .vtms_structs import *
from .vtk_exporter import *
from ..postprocess import write_fabric_mold_contact
from ..paths import get_output
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

@dataclass
class RigidMold:
    points: np.ndarray
    connections: np.ndarray
    point_ids: np.ndarray | None = None
    point_diameter: float = 0.0 # "cloud" diameter around the points for identifying contact, 0 indicates the points are surface points. 


def solve_fiber_mechanics_bvp(
    fabric: VTMSFabric,
    materials: list[VTMSFiberMaterial],
    boundary_conditions: list[DirichletBC | NeumannBC | PeriodicBC] | list[list[DirichletBC | NeumannBC| PeriodicBC]],
    solver_options: SolverOptions,
    pseudotime_iters: int = 1,
    plot_convergence: bool = False,
    blow_up_threshold: float = jnp.inf,
    filename_base: str | None = None,
    rigid_mold: RigidMold | None = None,
    pre_strain: float | None = None,
    contact_options: ContactParams | None = None,
    debug_info: DebugInfo | None = None,
):
    if debug_info is None:
        debug_info = NULL_DEBUG_INFO
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

    n_dofs_per_basis = fabric.points.shape[1]

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


    if pre_strain is not None:
        internal_state = jnp.full((
            connectivity_en.shape[0],
            get_quadrature(fe_type=fe_type)[0].shape[0],
            1
        ),pre_strain)
    element_batches = [
        ElementBatch(
            fe_type            = fe_type,
            n_dofs_per_basis   = n_dofs_per_basis,
            connectivity_en    = connectivity_en,
            constitutive_model = elastic_truss,
            material_params    = jnp.array(material_params),
            internal_state     = jnp.array(internal_state) if pre_strain is not None else None,
        )
    ]

    point_fiber_ids = np.concatenate(
        [
            np.full((fabric.fiber_offsets[i + 1] - fabric.fiber_offsets[i],), i)
            for i in range(fabric.fiber_offsets.shape[0] - 1)
        ]
    )
    # contact_search_radius = contact_options.contact_search_radius
    # surface_contact_alpha = getattr(contact_options, "surface_contact_alpha", None)
    # use_surface_contact = surface_contact_alpha is not None
    # if (contact_search_radius is None) == (surface_contact_alpha is None):
    #     raise ValueError(
    #         "ContactParams must define exactly one of contact_search_radius or "
    #         "surface_contact_alpha."
    #     )
    if contact_options.contact_search_alpha <= contact_options.M_to_D_ratio:
        raise ValueError(
            "contact_search_alpha must be greater than M_to_D_ratio."
        )
    if contact_options.M_to_D_ratio <= contact_options.C_to_D_ratio:
        raise ValueError(
            "M_to_D_ratio must be greater than C_to_D_ratio."
        )

    point_diameters = []
    global_fiber_i = 0
    for b_i in range(fabric.get_n_bundles()):
        for _ in range(fabric.get_n_fibers_in_bundle(b_i)):
            n_points = fabric.fiber_offsets[global_fiber_i + 1] - fabric.fiber_offsets[global_fiber_i]
            point_diameters.append(np.full((n_points,), fabric.get_diameter(b_i)))
            global_fiber_i += 1
    point_diameters = np.concatenate(point_diameters)

    if rigid_mold is not None:
        point_diameters = np.concatenate([point_diameters, np.full((rigid_mold.points.shape[0],),rigid_mold.point_diameter)])

        point_fiber_ids = np.concatenate([point_fiber_ids,rigid_mold.point_ids])
        element_batches+= [
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=n_dofs_per_basis,
                connectivity_en=rigid_mold.connections + fabric.points.shape[0],
                constitutive_model=elastic_truss,
                material_params=jnp.array([materials[0].E,materials[0].A]),
            )
        ]

    def assemble_vertices_vd(fabric, rigid_mold):
        fabric_n = fabric.points.shape[0]
        if rigid_mold is None:
            return fabric.points, fabric_n, None
        mold_n = rigid_mold.points.shape[0]
        vertices_vd = np.vstack([fabric.points, rigid_mold.points])
        return vertices_vd, fabric_n, slice(fabric_n, fabric_n + mold_n)
    vertices_vd, fabric_n, mold_slice = assemble_vertices_vd(fabric,rigid_mold)

    contact_fe_type = fe_type

    self_adjacency_block = contact_options.self_adjacency_block

    contact_E_c = contact_options.D_stiffness_to_E_ratio * np.max(material_params[:,0])
    contact_A = np.max(material_params[:,1])
    contact_E_min = contact_options.M_stiffness_to_E_ratio * np.max(material_params[:,0])
    # legacy_contact_params = None
    # if not use_surface_contact:
    #     legacy_contact_params = jnp.array([
    #         contact_E_c,
    #         contact_A,
    #         contact_search_radius,
    #         fabric.get_diameter(0),
    #         contact_options.M_to_D_ratio * fabric.get_diameter(0),
    #         contact_E_min,
    #     ])  # E_c, A, search_radius, D, M, E_min

    contact_output_params = {
        'self_adjacency_block': self_adjacency_block,
        'contact_search_radius': contact_options.contact_search_alpha,
        'point_diameters': point_diameters,
        'surface_contact_alpha': contact_options.contact_search_alpha,
    }

    def contact_pair_generator(u_ref) -> list[ElementBatch] | None:
        if u_ref is None:
            u_ref = jnp.zeros((vertices_vd.shape[0]*vertices_vd.shape[1],))
        contact_cells = contact_batch(
            points=vertices_vd + np.array(u_ref).reshape(vertices_vd.shape),
            point_fiber_ids=point_fiber_ids,
            adjacency_block=self_adjacency_block,
            point_diameters=point_diameters,
            search2radius_ratio=contact_options.contact_search_alpha,
        )
        if contact_cells.shape[0] == 0: return []

        point_radii = 0.5 * point_diameters
        contact_material_params = np.column_stack([
            np.full((contact_cells.shape[0],), contact_E_c),
            np.full((contact_cells.shape[0],), contact_A),
            point_radii[contact_cells[:,0]],
            point_radii[contact_cells[:,1]],
            np.full((contact_cells.shape[0],), contact_options.M_to_D_ratio),
            np.full((contact_cells.shape[0],), contact_options.C_to_D_ratio),
            np.full((contact_cells.shape[0],), contact_options.contact_search_alpha),
            np.full((contact_cells.shape[0],), contact_E_min),
        ])
        return [
            ElementBatch(
                fe_type=contact_fe_type,
                n_dofs_per_basis=n_dofs_per_basis,
                connectivity_en=contact_cells,
                constitutive_model=contact_options.contact_constitutive_model,
                material_params=jnp.array(contact_material_params),
            )
        ]

    if filename_base is not None:
        write_vtk(fabric,get_output(filename=f"{filename_base}_0.vtk"))
        write_fabric_mold_contact(
            fabric = fabric,
            mold = rigid_mold,
            filename = get_output(filename = f"{filename_base}_wireframe_0.vtk"),
            contact_params = contact_output_params,
        )

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
        debug_info.begin_stage(
            time_step=i,
            nonlinear_solve=0,
            linear_solve=0,
            current_stage=DebugOutputStage.TIME_STEP,
        )

        u_truss, residual_truss, element_batches_truss = solve_bvp(
            element_residual_func=linear_truss_residual,
            vertices_vd=vertices_vd,
            u_0_g=None if i==0 else u_truss,
            element_batches=element_batches,
            boundary_conditions=boundary_conditions[i],
            solver_options=solver_options,
            plot_convergence=plot_convergence,
            contact_batch_generator=contact_pair_generator,
            debug_info=debug_info,
            time_step = i,
        )
        debug_info.begin_stage(
            time_step=i+1,
            nonlinear_solve=0,
            linear_solve=0,
            current_stage=DebugOutputStage.TIME_STEP,
        )

        if debug_info.contains(DebugOutputQuantities.NODE_SOLUTION):
            debug_info.output(
                DebugOutputQuantities.NODE_SOLUTION,
                "u",
                u_truss.reshape((-1,fabric.points.shape[1]))
            )
        if debug_info.contains(DebugOutputQuantities.NODE_RESIDUAL):
            debug_info.output(
                DebugOutputQuantities.NODE_RESIDUAL,
                "residual",
                residual_truss.reshape((-1,fabric.points.shape[1]))
            )

        # u_truss = u_truss.reshape((-1,fabric.points.shape[1]))
        print(f"\nmax(||u||) = {np.linalg.norm(u_truss.reshape((-1,fabric.points.shape[1])),axis=1).max()}\n")
        # fabric.points = fabric.points + np.array(u_truss)

        if filename_base is not None:
            temp_fab = deepcopy(fabric)
            temp_fab.points += np.array(u_truss.reshape((-1,temp_fab.points.shape[1]))[:fabric_n,:])
            temp_mold = deepcopy(rigid_mold)
            if rigid_mold is not None:
                temp_mold.points +=  np.array(u_truss.reshape((-1,temp_mold.points.shape[1]))[fabric_n:,:])
            write_vtk(temp_fab,get_output(filename=f"{filename_base}_{i+1}.vtk"))
            write_fabric_mold_contact(
                fabric = temp_fab,
                mold = temp_mold,
                filename = get_output(filename = f"{filename_base}_wireframe_{i+1}.vtk"),
                contact_params = contact_output_params,
            )
        if jnp.isnan(u_truss).any() or jnp.isinf(u_truss).any() or np.linalg.norm(u_truss.reshape((-1,fabric.points.shape[1])),axis=1).max()>blow_up_threshold:
            raise RuntimeError(f"Nonlinear solve diverged: displacement magnitude exceeded threshold ({blow_up_threshold})")

    return u_truss, residual_truss, element_batches_truss
