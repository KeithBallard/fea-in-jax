from igl import adjacency_list
from fe_jax.contact import ContactPreprocessConfig
from .setup import *
from .utils import *
from .solve_cg import cg as cg_w_info
from .sparse_matrix import *
from .sparse_linear_solve import *
from .constraints import *
from .constraint_system import *
from .dof_enumeration import *
from .boundary_conditions import *
from .load_system import *
from . import contact
from .debug_outputs import *

import jax.numpy as jnp
import jax
import jax.experimental.sparse as jsparse

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any

from flax import struct


@struct.dataclass
class ElementBatch:
    """
    Describes a batch of elements. Passed into solve_bvp()
    """

    # Defines the type of finite element formulation (basis functions and quadrature) for the
    # elements in this batch.
    fe_type: FiniteElementType
    # Number of degrees of freedom per basis function (typically also per a node)
    n_dofs_per_basis: int
    # List of vertex indices for each element (refers to list of vertices passed to solve_bvp(),
    # not internal batch numbering)
    connectivity_en: np.ndarray[Any, np.dtype[np.uint64]]
    # A callable constitutive model for the batch of elements, which is passed along to the residual
    # function as an argument to perform the residual calculation.
    constitutive_model: Callable
    # Array can 1, 2, or 3 dimensions depending on whether parameters are defined 1) the same
    # for all quad points / elements in the batch [shape should be (M,)], 2) the same for all quad points
    # but varying across elements [shape should be (E, M)], or 3) varying across each quad point /
    # element in the batch [shape should be (E, Q, M)], respectively.
    material_params: jnp.ndarray
    # Array can 2 or 3 dimensions depending on whether state variables are defined 1) the same for
    # all quad points but varying across elements [shape should be (E, I)], or 3) varying across
    # each quad point / element in the batch [shape should be (E, Q, I)], respectively.
    internal_state: jnp.ndarray | None = None

    def __post_init__(self):
        Q = get_quadrature(fe_type=self.fe_type)[0].shape[0]
        if len(self.material_params.shape) == 2:
            # Dimensions should be (E, M)
            assert (
                self.material_params.shape[0] == self.connectivity_en.shape[0]
            ), f"`material_params` had dimension 2, which means the shape should be (E, M). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `material_params.shape[0]` ({self.material_params.shape[0]})"
        elif len(self.material_params.shape) == 3:
            # Dimensions should be (E, Q, M)
            assert (
                self.material_params.shape[0] == self.connectivity_en.shape[0]
            ), f"`material_params` had dimension 3, which means the shape should be (E, Q, M). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `material_params.shape[0]` ({self.material_params.shape[0]})"
            assert (
                self.material_params.shape[1] == Q
            ), f"`material_params` had dimension 3, which means the shape should be (E, Q, M). However, `fe_type` results in Q = {Q}, which did not match `material_params.shape[1]` ({self.material_params.shape[1]})"
        if self.internal_state is not None:
            if len(self.internal_state.shape) == 2:
                # Dimensions should be (E, I)
                assert (
                    self.internal_state.shape[0] == self.connectivity_en.shape[0]
                ), f"`internal_state` had dimension 2, which means the shape should be (E, I). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `internal_state.shape[0]` ({self.internal_state.shape[0]})"
            elif len(self.internal_state.shape) == 3:
                # Dimensions should be (E, Q, M)
                assert (
                    self.internal_state.shape[0] == self.connectivity_en.shape[0]
                ), f"`internal_state` had dimension 3, which means the shape should be (E, Q, I). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `internal_state.shape[0]` ({self.internal_state.shape[0]})"
                assert (
                    self.internal_state.shape[1] == Q
                ), f"`internal_state` had dimension 3, which means the shape should be (E, Q, I). However, `fe_type` results in Q = {Q}, which did not match `internal_state.shape[1]` ({self.internal_state.shape[1]})"


class MaterialPropertyArrayType(Enum):
    EQM = 3  # Unique set per quad point in each element
    EM = 2  # Unique set per element
    M = 1  # Same set for the entire element batch


class QuadratureArrayType(Enum):
    EQ = 2  # Unique quadrature per element
    Q = 1  # Same quadrature for the entire element batch


@struct.dataclass
class ElementBatchCollection:
    """
    Holds information about a collection of batches of elements in a form that is ameniable to JIT.
    """

    # --- Batch shape information (numpy / static to support JIT) ---

    # Dimensionality of mesh (1D, 2D, or 3D)
    # Note: static, not traced
    D: int = struct.field(pytree_node=False)
    # Number of batches
    # Note: static, not traced
    B: int = struct.field(pytree_node=False)
    # Number of elements for each batch
    # Note: static, not traced
    E: tuple[int, ...] = struct.field(pytree_node=False)
    # Number of nodes per element for each batch
    # Note: static, not traced
    N: tuple[int, ...] = struct.field(pytree_node=False)
    # Number of degrees of freedom (unknowns) per a node for each batch
    # Note: static, not traced
    U: tuple[int, ...] = struct.field(pytree_node=False)
    # Number of quadrature points per an element for each batch
    # Note: static, not traced
    Q: tuple[int, ...] = struct.field(pytree_node=False)
    # Dimensionality of parametric coordinate system for each batch
    # Note: static, not traced
    P: tuple[int, ...] = struct.field(pytree_node=False)
    # Number of material parameters required for each batch (at a point)
    # Note: static, not traced
    M: tuple[int, ...] = struct.field(pytree_node=False)
    # Number of internal state variables required for each batch (at a point)
    # Note: static, not traced
    I: tuple[int, ...] = struct.field(pytree_node=False)

    # --- Mesh / property / state information ---

    # Unravelled coordinates of all nodes for all elements across all batches, shape=(sum(E*N*D),)
    x: jnp.ndarray
    # Unravelled indices of nodes for all elements across all batches, shape=(sum(E*N),)
    connectivity: jnp.ndarray
    # Unravelled material parameters for all batches, shape depends on types for each batch
    material_params: jnp.ndarray
    # Unravelled internal state variables (ISV) for all batches, shape=(sum(E*Q*I),)
    internal_state: jnp.ndarray

    # --- Quadrature and basis function information ---

    # Unravelled quadrature point coordinates for all batches, shape depends on types
    xi: jnp.ndarray
    # Unravelled quadrature point weights for all batches, shape depends on types
    weights: jnp.ndarray
    # Unravelled basis functions evaluated at quad points for all batches, shape depends on types
    phi: jnp.ndarray
    # Unravelled derivative of basis functions (w.r.t. parametric coordinates) evaluated at quad
    # points for all batches, shape depends on types
    dphi_dxi: jnp.ndarray

    # --- Degree of freedom enumeration ---

    # Key information for degree-of-freedom numbering (enumeration)
    dof_enumeration: DofEnumeration
    # Unravelled rank DoF map for each element for all batches, length depends on the element types
    # for each batch.
    # NOTE: not needed at this time since DoF enumeration is simply tied to node numbers (until
    # XFEM is implemented).
    # element_to_rank_maps: jnp.ndarray
    # Offset for each batch into `element_to_rank_maps`, shape=(B+1,)
    # NOTE: not needed at this time since DoF enumeration is simply tied to node numbers (until
    # XFEM is implemented).
    # element_to_rank_offsets: jnp.ndarray

    # --- Callable functions ---

    # Constitutive model for each batch, length=B
    # Note: static, not traced
    constitutive_models: tuple[jax.tree_util.Partial, ...] = struct.field(
        pytree_node=False
    )

    # --- Offsets / sizes into expanded arrays for slicing ---

    # Element-node offset for each batch (used to index into `x` and `connectivity`), shape=(B+1,)
    EN_offsets: jnp.ndarray
    # Type of material_params for each batch. For each batch, the shape of an array can
    #  be one of three type:
    # 1) (E*Q*M,) if every quad point in every element has a unique set of material parameters
    # 2) (E*M,) if every element has a unique set of material parameters
    # 3) (M,) if each batch has a unique set of material parameters
    # Note: static, not traced
    material_params_types: list[MaterialPropertyArrayType] = struct.field(
        pytree_node=False
    )
    # Offset for each batch into `material_params`, shape=(B+1,)
    material_params_offsets: jnp.ndarray
    # Size of each batch stored in `material_params`, shape=(B,)
    material_params_sizes: tuple[int, ...] = struct.field(pytree_node=False)
    # Offset for each batch into `internal_state`, shape=(B+1,)
    internal_state_offsets: jnp.ndarray
    # Size of each batch stored in `internal_state`, shape=(B,)
    internal_state_sizes: tuple[int, ...] = struct.field(pytree_node=False)

    # Type of quadrature / basis for each batch, which can be either same quadrature / basis
    # for the entire batch or a different set of quad points / basis for each element.
    # Note: static, not traced
    quadrature_types: tuple[QuadratureArrayType, ...] = struct.field(pytree_node=False)
    # Offset for each batch into `xi`, shape=(B+1,)
    xi_offsets: jnp.ndarray
    # Offset for each batch into `weights`, shape=(B+1,)
    weights_offsets: jnp.ndarray
    # Offset for each batch into `phi`, shape=(B+1,)
    phi_offsets: jnp.ndarray
    # Offset for each batch into `dphi_dxi`, shape=(B+1,)
    dphi_dxi_offsets: jnp.ndarray

    @partial(jax.jit, static_argnames="i")
    def get_connectivity(self, i: int) -> jnp.ndarray:
        """
        Retrieves the (reshaped) `connectivity` array for batch i

        Args:
            i: Batch index

        Returns:
            out: Array of floats with shape (E, N)
        """
        return jax.lax.dynamic_slice(
            self.connectivity,
            start_indices=(self.EN_offsets[i],),
            slice_sizes=(self.E[i] * self.N[i],),
        ).reshape((self.E[i], self.N[i]))

    @partial(jax.jit, static_argnames="i")
    def get_material_params(self, i: int) -> jnp.ndarray:
        """
        Retrieves the (reshaped) `material_parameters` array for batch i

        Args:
            i: Batch index

        Returns:
            out: Array of floats with shape (E, Q, M), (E, M), or (M,) depending on material
                properties array type
        """
        match self.material_params_types[i]:
            case MaterialPropertyArrayType.EQM:
                return jax.lax.dynamic_slice(
                    self.material_params,
                    start_indices=(self.material_params_offsets[i],),
                    slice_sizes=(self.E[i] * self.Q[i] * self.M[i],),
                ).reshape((self.E[i], self.Q[i], self.M[i]))
            case MaterialPropertyArrayType.EM:
                return jax.lax.dynamic_slice(
                    self.material_params,
                    start_indices=(self.material_params_offsets[i],),
                    slice_sizes=(self.E[i] * self.M[i],),
                ).reshape((self.E[i], self.M[i]))
            case _:  # M
                return jax.lax.dynamic_slice(
                    self.material_params,
                    start_indices=(self.material_params_offsets[i],),
                    slice_sizes=(self.M[i],),
                ).reshape((self.M[i],))

    @partial(jax.jit, static_argnames="i")
    def get_internal_state(self, i: int) -> jnp.ndarray:
        """
        Retrieves the (reshaped) `internal_state` array for batch

        Args:
            i: Batch index

        Returns:
            out: Array of floats with shape (E, Q, I)
        """
        return jax.lax.dynamic_slice(
            self.internal_state,
            start_indices=(self.internal_state_offsets[i],),
            slice_sizes=(self.E[i] * self.Q[i] * self.I[i],),
        ).reshape((self.E[i], self.Q[i], self.I[i]))

    @partial(jax.jit, static_argnames="i")
    def get_x(self, i: int) -> jnp.ndarray:
        """
        Retrieves the (reshaped) `x` array for batch i

        Args:
            i: Batch index

        Returns:
            out: Array of floats with shape (E, N, D)
        """
        return jax.lax.dynamic_slice(
            self.x,
            start_indices=(self.D * self.EN_offsets[i],),
            slice_sizes=(self.E[i] * self.N[i] * self.D,),
        ).reshape((self.E[i], self.N[i], self.D))

    @partial(jax.jit, static_argnames="i")
    def get_weights(self, i: int) -> jnp.ndarray:
        """
        Retrieves the (reshaped) `weights` array for batch i

        Args:
            i: Batch index

        Returns:
            out: Array of floats with shape (E,) or (1,) depending on quadrature array type

        """
        match self.quadrature_types[i]:
            case QuadratureArrayType.EQ:
                return jax.lax.dynamic_slice(
                    self.weights,
                    start_indices=(self.weights_offsets[i],),
                    slice_sizes=(self.E[i],),
                )
            case _:  # Q
                return jax.lax.dynamic_slice(
                    self.weights,
                    start_indices=(self.weights_offsets[i],),
                    slice_sizes=(1,),
                )

    @partial(jax.jit, static_argnames="i")
    def get_dphi_dxi(self, i: int) -> jnp.ndarray:
        """
        Retrieves the (reshaped) `dphi_dxi` array for batch i.

        Args:
            i: Batch index

        Returns:
            out: Array of floats with shape (E, Q, N, P) or (Q, N, P) depending on quadrature
                array type
        """
        match self.quadrature_types[i]:
            case QuadratureArrayType.EQ:
                return jax.lax.dynamic_slice(
                    self.dphi_dxi,
                    start_indices=(self.dphi_dxi_offsets[i],),
                    slice_sizes=(self.E[i] * self.Q[i] * self.N[i] * self.P[i],),
                ).reshape(self.E[i], self.Q[i], self.N[i], self.P[i])
            case _:  # Q
                return jax.lax.dynamic_slice(
                    self.dphi_dxi,
                    start_indices=(self.dphi_dxi_offsets[i],),
                    slice_sizes=(self.Q[i] * self.N[i] * self.P[i],),
                ).reshape(self.Q[i], self.N[i], self.P[i])

    @partial(jax.jit, static_argnames="i")
    def get_dof_map(self, i: int) -> jnp.ndarray:
        """
        Returns the element degree of freedom map, which maps from a vector for the element to
        the DoF numbering.

        NOTE: if distributed computing is introduced (via MPI), we will need to distinguish
        between `rank` and `global` enumerations.

        Args:
            i: Batch index

        Returns:
            out: Array of integers with shape (E, N * U)
        """
        connectivity_en = self.get_connectivity(i)
        # Assumes each node has `U` number of DoFs and DoFs are enumerated following node numbering
        return jnp.vstack(
            [(self.U[i] * connectivity_en + j).ravel() for j in range(self.U[i])],
            dtype=jnp.int64,
        ).T.reshape((self.E[i], self.N[i] * self.U[i]))

    @property
    def is_homogeneous(self) -> bool:
        """
        Returns True if all batches in the collection have the same shape/type properties
        (N, U, Q, P, M, I, and material/quadrature types).
        """
        if self.B <= 1:
            return True

        # Check integer tuples
        check_attrs = ["N", "U", "Q", "P", "M", "I"]
        for attr in check_attrs:
            val = getattr(self, attr)
            if not all(v == val[0] for v in val):
                return False

        # Check type lists/tuples
        if not all(t == self.quadrature_types[0] for t in self.quadrature_types):
            return False

        if not all(
            t == self.material_params_types[0] for t in self.material_params_types
        ):
            return False

        return True


def batch_to_collection(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    dof_enumeration: DofEnumeration,
) -> ElementBatchCollection:
    """
    Converts a list of ElementBatch's to a BatchCollection, which is ameniable to JIT operations.
    """
    E = tuple((b.connectivity_en.shape[0] for b in element_batches))
    N = tuple((b.connectivity_en.shape[1] for b in element_batches))
    U = tuple((b.n_dofs_per_basis for b in element_batches))
    Q = tuple((get_quadrature(fe_type=b.fe_type)[0].shape[0] for b in element_batches))
    M = tuple((b.material_params.shape[-1] for b in element_batches))
    I = tuple(
        (
            b.internal_state.shape[-1] if b.internal_state is not None else 0
            for b in element_batches
        )
    )

    xi_bqp, W_bq = zip(*[get_quadrature(fe_type=b.fe_type) for b in element_batches])
    phi_bqn, dphi_dxi_bqnp = zip(
        *[
            eval_basis_and_derivatives(fe_type=b.fe_type, xi_qp=xi_bqp[i])
            for i, b in enumerate(element_batches)
        ]
    )

    x_bend = [
        mesh_to_jax(vertices=vertices_vd, cells=b.connectivity_en).ravel()
        for b in element_batches
    ]

    return ElementBatchCollection(
        # --- Batch shape information (numpy / static to support JIT) ---
        D=vertices_vd.shape[1],
        B=len(element_batches),
        E=E,
        N=N,
        U=U,
        Q=Q,
        M=M,
        P=tuple([xi_qp.shape[-1] for xi_qp in xi_bqp]),
        I=I,
        # --- Mesh / property / state information ---
        x=jnp.hstack([x_end.ravel() for x_end in x_bend]),
        connectivity=jnp.hstack(
            [b.connectivity_en.ravel() for b in element_batches], dtype=jnp.int64
        ),
        material_params=jnp.hstack(
            [b.material_params.ravel() for b in element_batches]
        ),
        internal_state=jnp.hstack(
            [
                (
                    b.internal_state.ravel()
                    if b.internal_state is not None
                    else jnp.zeros(shape=(E[i], Q[i], I[i])).ravel()
                )
                for i, b in enumerate(element_batches)
            ]
        ),
        # --- Quadrature and basis function information ---
        xi=jnp.hstack([xi_qp.ravel() for xi_qp in xi_bqp]),
        weights=jnp.hstack([W_q.ravel() for W_q in W_bq]),
        phi=jnp.hstack([phi_qn.ravel() for phi_qn in phi_bqn]),
        dphi_dxi=jnp.hstack([dphi_dxi_qnp.ravel() for dphi_dxi_qnp in dphi_dxi_bqnp]),
        # --- Degree of freedom enumeration ---
        dof_enumeration=dof_enumeration,
        # --- Callable functions ---
        constitutive_models=tuple(
            [jax.tree_util.Partial(b.constitutive_model) for b in element_batches]
        ),
        # --- Offsets / sizes into expanded arrays for slicing ---
        EN_offsets=jnp.hstack(
            [jnp.array([0]), jnp.cumsum(jnp.array(E) * jnp.array(N))]
        ),
        material_params_types=[
            MaterialPropertyArrayType(len(b.material_params.shape))
            for b in element_batches
        ],
        material_params_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(
                    jnp.array([b.material_params.size for b in element_batches])
                ),
            ]
        ),
        material_params_sizes=tuple([b.material_params.size for b in element_batches]),
        internal_state_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(
                    jnp.array(
                        [
                            (
                                b.internal_state.size
                                if b.internal_state is not None
                                else 0
                            )
                            for b in element_batches
                        ]
                    )
                ),
            ]
        ),
        internal_state_sizes=tuple(
            [
                b.internal_state.size if b.internal_state is not None else 0
                for b in element_batches
            ]
        ),
        quadrature_types=tuple([QuadratureArrayType.Q for b in element_batches]),
        xi_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(jnp.array([xi_qp.size for xi_qp in xi_bqp])),
            ]
        ),
        weights_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(jnp.array([W_q.size for W_q in W_bq])),
            ]
        ),
        phi_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(jnp.array([phi_qn.size for phi_qn in phi_bqn])),
            ]
        ),
        dphi_dxi_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(
                    jnp.array([dphi_dxi_qnp.size for dphi_dxi_qnp in dphi_dxi_bqnp])
                ),
            ]
        ),
    )


@partial(jax.jit, static_argnames="n_vertices")
def _calculate_jacobian_unique_nnz(
    n_vertices: int,
    ebc: ElementBatchCollection,
):
    """
    Returns the number of non-zeros in the Jacobian for a collection of batches of elements,
    ignoring any effect of constraints on the sparsity pattern.
    """
    node_nnz_count = jnp.zeros((n_vertices,), dtype=jnp.int64)

    @partial(jax.jit, static_argnames="i")
    def jacobian_indices(i: int):
        dof_map = ebc.get_dof_map(i)
        cols, rows = jax.vmap(jnp.meshgrid)(dof_map, dof_map)
        return jnp.vstack([rows.ravel(), cols.ravel()]).T

    non_zero_indices = jnp.vstack([jacobian_indices(i) for i in range(ebc.B)])
    # Get the permutation that sorts the non-zero entries (sorted by row then col)
    perm = jnp.lexsort((non_zero_indices[:, 1], non_zero_indices[:, 0]))
    # Sort the non-zero indices
    non_zero_indices = non_zero_indices[perm]
    # An array of non_zero_indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(non_zero_indices, axis=0)
    # Boolean mask indicating if each (row, col) value is unique, shape=A.col.shape
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))
    return jnp.sum(uniq_mask)


@jax.jit
def _calculate_jacobian_batch_element_kernel(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    u_enu: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the element-level jacobian matrices for a batch of elements without any modification
    of the solution or residual to accomodate constraints.

    Parameters
    ----------
    element_residual_func: residual function emerging from weak form of governing equations
    constitutive_model: a partial function for the constitutive model
    u_enu: element node solution displacement array, ndarray[float, (E, N, U)]
    x_end: element node coordinate array, ndarray[float, (E, N, D)]
    dphi_dxi_qnp: basis function derivatives at quadrature points, ndarray[float, (Q, N, P)]
    W_q: quadrature weights, ndarray[float, (Q,)]
    material_params: material parameters for the batch
    internal_state: internal state variables for the batch

    Returns
    -------
    out: batch of element-level Jacobian matrices, ndarray[float, (E, N * U, N * U)]
    """

    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]
    U = u_enu.shape[2]

    # Note: reshaped to be (# elements, # dofs per element) so that the jacfwd produces a
    # (# dofs per element, # dofs per element) matrix for each element.
    # Assumption: # dofs per element is N * U
    u_et = u_enu.reshape(E, N * U)

    # Note: captures dphi_dxi_qnp, W_q, and constitutive_model
    @jax.jit
    def residual_kernel(u_t, x_nd, material_params, internal_state_qi):
        u_nd = u_t.reshape(N, D)
        R_nu = element_residual_func(
            u_nd=u_nd,
            x_nd=x_nd,
            dphi_dxi_qnp=dphi_dxi_qnp,
            W_q=W_q,
            material_params=material_params,
            internal_state_qi=internal_state_qi,
            constitutive_model=constitutive_model,
        )[0]
        return R_nu.reshape(N * U)

    J_ett = jax.vmap(
        jax.jacfwd(residual_kernel, argnums=0),
        in_axes=(
            0,
            0,
            None if material_params.ndim == 1 else 0,
            None if internal_state.ndim < 3 else 0,
        ),
    )(u_et, x_end, material_params, internal_state)

    assert J_ett.shape == (
        E,
        N * U,
        N * U,
    ), f"Expected shape {(E, N * U, N * U)}, but received {J_ett.shape}"

    return J_ett


@jax.jit
def _calculate_jacobian_coo_terms_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    dof_map_enu: jnp.ndarray,
    assembly_map: jsparse.BCSR,
    u_f: jnp.ndarray,
):
    u_enu = transform_global_unraveled_to_element_node(
        assembly_map, u_f, x_end.shape[0]
    )

    dof_map = dof_map_enu.reshape(x_end.shape[0], -1)
    # debug_print(dof_map)
    cols, rows = jax.vmap(jnp.meshgrid)(dof_map, dof_map)
    # debug_print(rows)
    # debug_print(cols)

    J_ett = _calculate_jacobian_batch_element_kernel(
        element_residual_func=element_residual_func,
        constitutive_model=constitutive_model,
        u_enu=u_enu,
        x_end=x_end,
        dphi_dxi_qnp=dphi_dxi_qnp,
        W_q=W_q,
        material_params=material_params,
        internal_state=internal_state,
    )
    # debug_print(J_ett)

    return (J_ett, rows, cols)


@partial(jax.jit, static_argnames="deb")
def calculate_jacobian_wo_constraints(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
    precomputed_jacobian_nnz: int,
    deb: DebugInfo,
):

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    J_bett, rows, cols = zip(
        *[
            _calculate_jacobian_coo_terms_batch(
                element_residual_func=element_residual_func,
                constitutive_model=ebc.constitutive_models[i],
                material_params=ebc.get_material_params(i),
                internal_state=ebc.get_internal_state(i),
                x_end=ebc.get_x(i),
                dphi_dxi_qnp=ebc.get_dphi_dxi(i),
                W_q=ebc.get_weights(i),
                dof_map_enu=ebc.get_dof_map(i),
                assembly_map=assembly_map_b[i],
                u_f=u_f,
            )
            for i in range(ebc.B)
        ]
    )

    if deb.contains(DebugOutputQuantities.ELEMENT_JACOBIAN):
        for i, J_ett in enumerate(J_bett):
            deb.batch_output(DebugOutputQuantities.ELEMENT_JACOBIAN, i, J_bett[i])

    J_ett = jnp.vstack(J_bett)
    rows = jnp.vstack(rows)
    cols = jnp.vstack(cols)

    # debug_print(J_ett)
    # debug_print(rows)
    # debug_print(cols)

    J_sparse_ff = jsparse.COO(
        (J_ett.ravel(), rows.ravel(), cols.ravel()),
        shape=(u_f.shape[0], u_f.shape[0]),
    )._sort_indices()

    J_sparse_ff = coo_sum_duplicates(
        J_sparse_ff, result_length=precomputed_jacobian_nnz
    )

    return J_sparse_ff


@jax.jit
def _calculate_jacobian_diag_batch_element_kernel(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    u_enu: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the element-level jacobian diagonal matrices for a batch of elements without
    any modification of the solution or residual to accomodate constraints.

    Parameters
    ----------
    element_residual_func: residual function emerging from weak form of governing equations
    constitutive_model: a partial function for the constitutive model
    u_enu: element node solution displacement array, ndarray[float, (E, N, U)]
    x_end: element node coordinate array, ndarray[float, (E, N, D)]
    dphi_dxi_qnp: basis function derivatives at quadrature points, ndarray[float, (Q, N, P)]
    W_q: quadrature weights, ndarray[float, (Q,)]
    material_params: material parameters for each element batch
    internal_state: internal state variables for each element batch

    Returns
    -------
    diag_J_et: batch of element-level Jacobian diagonal vectors, ndarray[float, (E, N * U)]
    """

    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]
    U = u_enu.shape[2]

    # Note: reshaped to be (# elements, # dofs per element) so that the jacfwd produces a
    # (# dofs per element, # dofs per element) matrix for each element.
    # Assumption: # dofs per element is N * U
    u_et = u_enu.reshape(E, N * U)

    # Note: captures dphi_dxi_qnp, W_q, and constitutive_model
    @jax.jit
    def residual_kernel(u_t, x_nd, material_params, internal_state):
        u_nd = u_t.reshape(N, D)
        R_nu = element_residual_func(
            u_nd=u_nd,
            x_nd=x_nd,
            dphi_dxi_qnp=dphi_dxi_qnp,
            W_q=W_q,
            material_params=material_params,
            internal_state_qi=internal_state,
            constitutive_model=constitutive_model,
        )[0]
        return R_nu.reshape(N * U)

    def diag_J(u_t, x_nd, material_params, internal_state):
        return jnp.diagonal(
            jax.jacfwd(residual_kernel, argnums=0)(
                u_t, x_nd, material_params, internal_state
            )
        )

    diag_J_vmap = jax.vmap(
        diag_J,
        in_axes=(
            0,
            0,
            None if material_params.ndim == 1 else 0,
            None if internal_state.ndim < 3 else 0,
        ),
    )

    diag_J_et = diag_J_vmap(u_et, x_end, material_params, internal_state)

    assert diag_J_et.shape == (
        E,
        N * U,
    ), f"Expected shape {(E, N * U)}, but received {diag_J_et.shape}"

    return diag_J_et


@jax.jit
def _calculate_jacobian_diag_coo_terms_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    dof_map_enu: jnp.ndarray,
    assembly_map: jsparse.BCSR,
    u_f: jnp.ndarray,
):
    u_enu = transform_global_unraveled_to_element_node(
        assembly_map, u_f, x_end.shape[0]
    )

    dof_map = dof_map_enu.reshape(x_end.shape[0], -1)
    # debug_print(dof_map)

    diag_J_et = _calculate_jacobian_diag_batch_element_kernel(
        element_residual_func=element_residual_func,
        constitutive_model=constitutive_model,
        u_enu=u_enu,
        x_end=x_end,
        dphi_dxi_qnp=dphi_dxi_qnp,
        W_q=W_q,
        material_params=material_params,
        internal_state=internal_state,
    )
    # debug_print(diag_J_et)

    return (diag_J_et, dof_map)


def calculate_jacobian_diag_wo_constraints(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
):

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    diag_J_et, indices = zip(
        *[
            _calculate_jacobian_diag_coo_terms_batch(
                element_residual_func=element_residual_func,
                constitutive_model=ebc.constitutive_models[i],
                material_params=ebc.get_material_params(i),
                internal_state=ebc.get_internal_state(i),
                x_end=ebc.get_x(i),
                dphi_dxi_qnp=ebc.get_dphi_dxi(i),
                W_q=ebc.get_weights(i),
                dof_map_enu=ebc.get_dof_map(i),
                assembly_map=assembly_map_b[i],
                u_f=u_f,
            )
            for i in range(ebc.B)
        ]
    )
    diag_J_et = jnp.vstack(diag_J_et).ravel()
    indices = jnp.vstack(indices).ravel()

    # debug_print(diag_J_et)
    # debug_print(indices)

    diag_J_f = jnp.zeros_like(u_f)
    diag_J_f = diag_J_f.at[indices].add(diag_J_et)

    return diag_J_f


@jax.jit
def _calculate_residual_wo_constraints_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    assembly_map: jsparse.BCSR,
    u_f: jnp.ndarray,
):
    # Extract shape constants needed for args
    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]

    assert (
        N == dphi_dxi_qnp.shape[1]
    ), f"Number of nodes per element {N} must match the number of basis functions {dphi_dxi_qnp.shape[1]}."

    u_enu = transform_global_unraveled_to_element_node(assembly_map, u_f, E)

    # A vmap'ed version of the element residual function that maps over the elements
    R_vmap = jax.vmap(
        element_residual_func,
        in_axes=(
            0,  # u_end -> u_nd
            0,  # x_end -> x_nd
            None,  # dphi_dxi_qnp
            None,  # W_q
            (
                None if material_params.ndim == 1 else 0
            ),  # material_params_eqm -> material_params_qm or material_params_em -> material_params_m
            (
                None if internal_state.ndim < 3 else 0
            ),  # internal_state_eqi -> internal_state_qi
            None,  # constitutive_model
        ),
    )

    R_enu, internal_state = R_vmap(
        u_enu,
        x_end,
        dphi_dxi_qnp,
        W_q,
        material_params,
        internal_state,
        constitutive_model,
    )

    return R_enu, internal_state


def calculate_residual_wo_constraints(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
):
    """
    Calculates the residual and updated internal state variables without any modification
    of the solution or residual to accomodate constraints.

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    u_f                   : current solution (displacement), ndarray[float, (V * D)]

    Returns
    -------
    R_f                     : residual vector evaluated at the solution, ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    """
    # TODO change the pattern to accept donated arrays to hold R_f and new_internal_state_beqi

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    result = [
        _calculate_residual_wo_constraints_batch(
            element_residual_func=element_residual_func,
            constitutive_model=ebc.constitutive_models[i],
            material_params=ebc.get_material_params(i),
            internal_state=ebc.get_internal_state(i),
            x_end=ebc.get_x(i),
            dphi_dxi_qnp=ebc.get_dphi_dxi(i),
            W_q=ebc.get_weights(i),
            assembly_map=assembly_map_b[i],
            u_f=u_f,
        )
        for i in range(ebc.B)
    ]  # for each item, 0: R_end, 1: internal_state

    R_f = jnp.zeros_like(u_f)
    for i in range(ebc.B):
        R_f += transform_element_node_to_global_unraveled_sum(
            assembly_map=assembly_map_b[i], v_en=result[i][0]
        )

    new_internal_state_beqi = [result[i][1] for i in range(ebc.B)]
    # TODO split this out into a separate call

    # NOTE here is an alternative implementation leveraging fori, but the index i is a traced
    # array and therefore cannot be used to index into the lists, such as a constitutive_model_b.
    # Keeping this implementation here to revisit for optimization.
    """
    def fori_body(i, R_f) -> jnp.ndarray:
        R_enu, internal_state = _calculate_residual_wo_dirichlet_batch(
            element_residual_func=element_residual_func,
            constitutive_model=constitutive_model_b[i],
            material_params=material_params_beqm[i],
            internal_state=internal_state_beqi[i],
            x_end=x_bend[i],
            dphi_dxi_qnp=dphi_dxi_bqnp[i],
            W_q=W_bq[i],
            assembly_map=assembly_map_b[i],
            u_f=u_f,
        )
        return R_f + transform_element_node_to_global_unraveled_sum(
            assembly_map=assembly_map_b[i], v_en=R_enu
        )

    R_f = jax.lax.fori_loop(
        lower=0, upper=B, body_fun=fori_body, init_val=jnp.zeros_like(u_f), unroll=True
    )
    """

    return R_f, new_internal_state_beqi


def calculate_residual_w_constraints(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
    constraints: ConstraintSystem,
    f_ext,
):
    """
    Compute the residual vector and updated internal state variables given the current
    solution and state information with constraints applied.

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    u_f                   : current solution (displacement), ndarray[float, (V * D)]
    constraints           : system of linear constraints (MultiPointConstraints and Dirichlet BCs)

    Returns
    -------
    R_f                     : residual vector evaluated at the solution with constraints applied,
                              ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    """
    # Note: this is neccessary to ensure the Jacobian is symmetric. Without this,
    # the autodiff would result in 0's on rows (except on the diagonal) for entries
    # corresponding to Dirichlet BC's, but the columns would be non-zero.
    # debug_print(u_f)
    u_f_w_constraints = constraints.apply_to_solution(u_f)
    # debug_print(u_f_w_constraints)

    R_f, new_internal_state_beqi = calculate_residual_wo_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f_w_constraints,
    )

    # Zero out terms corresponding to Dirichlet BCs and add (solution - what it should be) for those constrained DoFs.
    # This will ensure there will be a 1 on the diagonal of the Jacobian and also return the right residual.
    # debug_print(R_f)
    R_f = f_ext.apply_to_residual(R_f)
    R_f = constraints.apply_to_residual(R_f, u_f)
    # debug_print(R_f)

    return R_f, new_internal_state_beqi


def solve_nonlinear_step(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    jacobian_nnz: int,
    u_0_g: jnp.ndarray,
    constraints: ConstraintSystem,
    solver_options: SolverOptions,
    f_ext: LoadSystem,
    element_diagnostic_outputs: Callable | None = None,
):
    """
    Solve the linearized system of equations emerging from the governing equations.
    This can be used within an outer loop to solve linear PDEs across time steps with different
    boundary conditions or to solve a nonlinear problem (via Newton's method for example).

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    jacobian_nnz          : number of non-zeros in the Jacobian matrix
    u_0_g                 : initial solution guess, ndarray[float, (V * D)]
    constraints           : system of linear constraints (MultiPointConstraints and Dirichlet BCs)
    solver_options        : options for the linear and nonlinear solvers

    Returns
    -------
    u_f                     : solution (displacement), ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    R_f                     : residual vector evaluated at the solution, ndarray[float, (V * D)]
    relative_error          : final relative error (L2 norm)
    info                    : solver result information
    """

    # Helpful for debugging array shapes
    # """
    print(f"Global dimensionality : {ebc.D}")
    print(f"# of batches : {ebc.B}")
    for i in range(ebc.B):
        print(
            f"For batch {i}:\n\t",
            f"Number of elements : {ebc.E[i]}\n\t",
            f"Number of nodes / element : {ebc.N[i]}\n\t",
            f"Number of quadrature points : {ebc.Q[i]}\n\t",
            f"Parametric dimensionality: {ebc.P[i]}\n\t",
            f"Number of material parameters per quad point: {ebc.M[i]}",
        )
    # """

    # Function that produces (R(u), ISVs)
    residual_isv_func_w_constraints = lambda u_f: calculate_residual_w_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f,
        constraints=constraints,
        f_ext=f_ext,
    )

    # Function that produces R(u)
    residual_func_w_constraints = lambda u_f: residual_isv_func_w_constraints(u_f=u_f)[
        0
    ]

    # Function that produces J(u) without Dirichlet BCs and MPCs applied
    jacobian_func_wo_constraints = lambda u_f: calculate_jacobian_wo_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f,
        precomputed_jacobian_nnz=jacobian_nnz,
    )

    # Function that produces diag(J(u)) without Dirichlet BCs and MPCs applied
    jacobian_diag_func_wo_constraints = (
        lambda u_f: calculate_jacobian_diag_wo_constraints(
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            u_f=u_f,
        )
    )

    R_f, new_internal_state_beqi = residual_isv_func_w_constraints(u_f=u_0_g)
    initial_R_f_norm = jnp.linalg.norm(R_f)

    element_diagnostic_outputs()

    def while_cond(args) -> bool:
        nl_iteration, u_f, R_f, new_internal_state_beqi, info = args
        absolute_error = jnp.linalg.norm(R_f)
        relative_error = absolute_error / initial_R_f_norm
        jax.debug.print(
            "End of iteration {x} rel error {y}, abs error {z}",
            x=nl_iteration - 1,
            y=relative_error,
            z=absolute_error,
        )
        """
        jax.debug.print(
            "Convergence criteria: {} {} {}",
            nl_iteration < solver_options.nonlinear_max_iter,
            relative_error > solver_options.nonlinear_relative_tol,
            absolute_error > solver_options.nonlinear_absolute_tol,
        )
        """
        return (
            (nl_iteration < solver_options.nonlinear_max_iter)
            & (relative_error > solver_options.nonlinear_relative_tol)
            & (absolute_error > solver_options.nonlinear_absolute_tol)
        )

    print(
        "WARNING: If using a solver that requires a Jacobian, Dirichlet BCs are being applied but multi-point constraints are NOT."
    )

    def while_body(
        args: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo]:
        nl_iteration, u_f, R_f, new_internal_state_beqi, info = args

        delta_u, info = linear_solve(
            residual=Residual(
                function=jax.tree_util.Partial(residual_func_w_constraints),
                dirichlet_bcs_builtin=True,
            ),
            jacobian=Jacobian(
                function=jax.tree_util.Partial(jacobian_func_wo_constraints),
                dirichlet_bcs_builtin=False,
            ),
            jacobian_diagonal=JacobianDiagonl(
                function=jax.tree_util.Partial(jacobian_diag_func_wo_constraints),
                dirichlet_bcs_builtin=False,
            ),
            constraints=constraints,
            solver_options=solver_options,
            solver_info_0=info,
            check_consistency=False,
            x_0=u_f,
            f_ext=f_ext,
        )

        D = ebc.U[0]
        max_d = solver_options.max_linear_displacement
        # TODO this only words if delta_u ONLY consists of nodal values, if other global DOFs are present this, need to be update (both the norm and the reshape with D.
        max_u = jnp.max(jnp.linalg.norm(delta_u.reshape((-1, D)), axis=1))
        scale = jnp.minimum(1.0, max_d / jnp.maximum(1e-16, max_u))
        delta_u = delta_u * scale
        u_f = u_f + delta_u
        R_f = residual_isv_func_w_constraints(u_f=u_f)[0]

        return (
            nl_iteration + 1,
            u_f,
            R_f,
            new_internal_state_beqi,
            info.increment_nl_iteration(),
        )

    _, u_f, R_f, new_internal_state_beqi, info = jax.lax.while_loop(
        cond_fun=while_cond,
        body_fun=while_body,
        init_val=(
            0,
            u_0_g,
            R_f,
            new_internal_state_beqi,
            init_solver_info(solver_options),
        ),
    )

    absolute_error = jnp.linalg.norm(R_f)
    relative_error = absolute_error / initial_R_f_norm

    return (u_f, new_internal_state_beqi, R_f, relative_error, info)


def preprocess_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: Callable,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    contact_batch_generator: Callable | None = None,
    u_0_g: jnp.ndarray | None = None,
):
    """
    Converts information from a user-facing format to a JAX-ameniable format.
    """
    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []

    # For 1D problems, the vertices may be given as a 1D array, so we need to reshape it to a 2D array
    if vertices_vd.ndim == 1:
        vertices_vd = np.expand_dims(vertices_vd, axis=1)

    B = len(element_batches)
    V = vertices_vd.shape[0]
    D = vertices_vd.shape[1]
    U = element_batches[0].n_dofs_per_basis
    n_total_dofs = V * U + sum(global_values)

    if contact_batch_generator is not None:
        element_batches = [*element_batches, *contact_batch_generator(u_0_g)]
        # TODO print how many contact elements were discovered

    # Validate input
    assert D <= 3
    assert len(boundary_conditions) <= D * V
    for b in element_batches:
        assert b.connectivity_en.shape[1] <= V
    for b in element_batches:
        assert (
            b.n_dofs_per_basis == element_batches[0].n_dofs_per_basis
        ), "The current DoF enumeration algorithm requires that the number of DoFs per a basis support point be constant across batches."

    # Wrap the provided callable to be compatible with jit
    element_residual_func = jax.tree_util.Partial(element_residual_func)

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

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
    # print(ebc)

    assert (
        np.array(ebc.U) == ebc.U[0]
    ).all(), """The number of DoFs per a point (U) must be the same across all batches.
    To relax this constraint much of the infrastructure code in fea.py would have to be adapted to
    support varying number of DoFs per a batch.
    """

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Compute the anticipated number of non-zeros for the assembled Jacobian, which
    # is only needed for solvers that actually form the Jacobian in memory.
    # NOTE: we need a concrete value to specialize for JIT of other functions
    jacobian_nnz = int(_calculate_jacobian_unique_nnz(n_vertices=V, ebc=ebc))

    fixed_point_constraints, boundary_multipoint_constraints = (
        convert_boundary_conditions_to_constraints(
            boundary_conditions=boundary_conditions,
            vertices_vd=vertices_vd,
            dof_enumeration=dof_enumeration,
            n_solution_components=ebc.U[0],
            global_values=global_values,
        )
    )
    multipoint_constraints = consolidate_multipoint_constraints(
        fixed_point_constraints=fixed_point_constraints,
        multipoint_constraints=[
            *boundary_multipoint_constraints,
            *multipoint_constraints,
        ],
    )

    constraint_system = convert_constraints_to_system(
        fixed_point_constraints=fixed_point_constraints,
        multipoint_constraints=multipoint_constraints,
        n_total_dofs=n_total_dofs,
    )

    external_load = convert_boundary_conditions_to_external_load(
        boundary_conditions=boundary_conditions,
        vertices_vd=vertices_vd,
        dof_enumeration=dof_enumeration,
        n_solution_components=ebc.U[0],
        global_values=global_values,
    )

    f_ext = convert_external_load_to_system(external_load)

    return (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    )


def solve_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: Callable,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    solver_options: SolverOptions = SolverOptions(),
    plot_convergence: bool = False,
    profile_memory: bool = False,
    contact_batch_generator: Callable | None = None,
    element_diagnostic_outputs: Callable | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, list[ElementBatch]]:
    """
    Solve a boundary value problem for static linear elasticity.

    Parameters
    ----------
    vertices_vd          : vertices needed for all cells on the rank, ndarray[float, (V, D)]
    element_batches      : batch of elements for this rank
    element_residual_func: residual function emerging from weak form of governing equations
    dirichlet_bcs        : Dirichlet boundary conditions, list[DirichletConstraint]
    multipoint_constraints : Linear constraints between degrees of freedom, list[MultiPointConstraint]
    global_values        : Length of list indicates number of global solution vector-values that will
                           added to the global system (e.g. for periodic BCs). Each entry in the list
                           indicates the number of components for each vector-value.
    u_0_g                : initial guess for the solution, ndarray[float, (V * D)] or None (default, zeros will be used)
    solver_options       : options for the linear/nonlinear solvers
    plot_convergence     : indicates if the convergence history for the linear solver should be
                           plotted via matplotlib as a figure
    profile_memory       : indicates if GPU memory usage should be profiled, which will create *.prof
                           files in the current directory

    Returns
    -------
    u               : solution (displacement), ndarray[float, (V * D)]
    R               : residual vector evaluated at the solution, ndarray[float, (V * D)]
    element_batches : element batches with updated internal state variables
    """
    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
        contact_batch_generator=contact_batch_generator,
        u_0_g=u_0_g,
    )

    n_total_dofs = vertices_vd.shape[0] * ebc.U[0] + sum(global_values)

    # If an initial guess was not provided, then use zeros
    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(n_total_dofs,))
    else:
        assert u_0_g.shape == (n_total_dofs,)

    inner_solve = solve_nonlinear_step
    if ebc.is_homogeneous:
        print("Batches are homogeneous, using JIT compilation for solve_linear_step")
        inner_solve = jax.jit(
            solve_nonlinear_step,
            # donate_argnames="internal_state_beqi",
            static_argnames=["solver_options", "jacobian_nnz"],
        )

    # capture memory usage before
    if profile_memory:
        start_memory_profile("solve_linear_step")

    u, internal_state_beqi, residual, relative_error, info = inner_solve(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        jacobian_nnz=jacobian_nnz,
        u_0_g=u_0_g,
        constraints=constraint_system,
        solver_options=solver_options,
        f_ext=f_ext,
    )

    # Update internal state variables for the element batches
    # TODO need to update
    # for i, b in enumerate(element_batches):
    #    b.internal_state = internal_state_beqi[i]

    # capture memory usage after and analyze
    if profile_memory:
        u.block_until_ready()
        stop_memory_profile("solve_linear_step")

    if info.cumulative_linear_iterations > 0:
        print(
            f"Cumulative # of linear solver iterations: {info.cumulative_linear_iterations}"
        )
        if plot_convergence:
            plot_solver_info(opts=solver_options, info=info)

    return (u, residual, element_batches)
