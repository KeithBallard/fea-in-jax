from fe_jax.helper import *
import jax
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

points = np.array([[0,0,0],[1,0,0]])
connections = np.array([[0,1]])
m_par = jnp.array([1,1])

fe_type = FiniteElementType(
    cell_type=CellType.interval,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=3,
)
el = [ElementBatch(
    fe_type = fe_type,
    n_dofs_per_basis = points.shape[1],
    connectivity_en=connections,
    constitutive_model=elastic_truss,
    material_params=m_par,
    internal_state = jnp.array([]),
)]

def get_ebc(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial = linear_truss_residual,
    # boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    # multipoint_constraints: List[MultiPointConstraint] | None = None,
    # global_values: List[int] | None = None,
    # contact_batch_generator: Callable | None = None,
    # u_0_g: jnp.ndarray | None = None,
):
    """
    Converts information from a user-facing format to a JAX-ameniable format.
    """
    # if boundary_conditions is None:
    #     boundary_conditions = []
    # if multipoint_constraints is None:
    #     multipoint_constraints = []
    global_values = []

    # For 1D problems, the vertices may be given as a 1D array, so we need to reshape it to a 2D array
    if vertices_vd.ndim == 1:
        vertices_vd = np.expand_dims(vertices_vd, axis=1)

    B = len(element_batches)
    V = vertices_vd.shape[0]
    D = vertices_vd.shape[1]
    U = element_batches[0].n_dofs_per_basis
    n_total_dofs = V * U + sum(global_values)

    # Validate input
    assert D <= 3
    for b in element_batches:
        assert b.connectivity_en.shape[1] <= V
    for b in element_batches:
        assert (
            b.n_dofs_per_basis == element_batches[0].n_dofs_per_basis
        ), "The current DoF enumeration algorithm requires that the number of DoFs per a basis support point be constant across batches."

    # Structures for mapping between cell-level arrays and global arrays
    # assembly_map_b = [
    #     mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
    #     for b in element_batches
    # ]

    # Enumerate degrees of freedom
    # NOTE: this currently assumes that the element_batches contains ALL elements
    # that will exist on this rank for the respective solve. If this is not the case,
    # we will need to construct the enumeration at a point where all element information
    # is known and pass it into this function.
    # NOTE assertion above ensures U is constant across batches
    dof_enumeration = DofEnumeration(
        n_owned_elements=sum([b.connectivity_en.shape[0] for b in element_batches]),
        n_owned_dofs=n_total_dofs,
        n_local_ghost_dofs=0,
        n_exclusive_ghost_dofs=0,
        n_free_global_dofs=sum(global_values),
        free_global_dof_rank_begin=V * U,
        owned_global_dof_begin=0,
        owned_global_dof_end=n_total_dofs,
        rank_to_global_map=jnp.arange(n_total_dofs),
    )

    # Convert element batch information into something ameniable to JAX transforms like JIT
    ebc = batch_to_collection(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        dof_enumeration=dof_enumeration,
    )
    return ebc
    # print(ebc)

ebc = get_ebc(vertices_vd=points,element_batches=el)
dphi_dxi = ebc.get_dphi_dxi(0)
W_q = ebc.get_weights(0)


def get_residuals(x,u,internal_state_qi = jnp.zeros((dphi_dxi.shape[0], 2))):
    R_stiff = stiffness_residual(
        u_nd = u,
        x_nd = x,
        dphi_dxi_qnp=dphi_dxi,
        W_q = W_q,
        material_params=m_par,
        internal_state_qi=internal_state_qi,
        constitutive_model=elastic_truss,
    )
    R_linear_truss = linear_truss_residual(
        u_nd = u,
        x_nd = x,
        dphi_dxi_qnp=dphi_dxi,
        W_q = W_q,
        material_params=m_par,
        internal_state_qi=internal_state_qi,
        constitutive_model=elastic_truss,
    )
    return R_stiff, R_linear_truss

points = jnp.array([[0,0,0],[1,0,0]])
u1 = jnp.array([[0.1,0.1,0.1],[0.05,0.1,0.2]])
u2 = jnp.array([[0.12,0.05,0.09],[0.2,0.3,0.4]])
S1, L1 = get_residuals(x=points[connections[0]],u=u1)
#S2, _ = get_residuals(x=points[connections[0]],u=u1+u2)
#points += u1
#S2, L2 = get_residuals(x=points[connections[0]],u=u2,internal_state_qi=L1[1])
print(S1[0],L1[0])
