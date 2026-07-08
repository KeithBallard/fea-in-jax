from .dof_enumeration import *
from .element_batch import *
from .setup import *

import jax
import jax.numpy as jnp

from flax import struct
from enum import Enum
from functools import partial
from typing import Callable, Any


class MaterialPropertyArrayType(Enum):
    EQM = 3  # Unique set per quad point in each element
    EM = 2  # Unique set per element
    M = 1  # Same set for the entire element batch


class QuadratureArrayType(Enum):
    EQ = 2  # Unique quadrature per element
    Q = 1  # Same quadrature for the entire element batch


@partial(jax.jit, static_argnames=["i", "E", "N"])
def ebc_get_connectivity(
    i: int,
    E: tuple[int, ...],
    N: tuple[int, ...],
    connectivity: jnp.ndarray,
    EN_offsets: jnp.ndarray,
) -> jnp.ndarray:
    """
    Retrieves the (reshaped) `connectivity` array for batch i
    """
    return jax.lax.dynamic_slice(
        connectivity,
        start_indices=(EN_offsets[i],),
        slice_sizes=(E[i] * N[i],),
    ).reshape((E[i], N[i]))


@partial(jax.jit, static_argnames=["i", "E", "N", "U"])
def ebc_get_dof_map(
    i: int,
    E: tuple[int, ...],
    N: tuple[int, ...],
    U: tuple[int, ...],
    connectivity: jnp.ndarray,
    EN_offsets: jnp.ndarray,
) -> jnp.ndarray:
    """
    Returns the element degree of freedom map, which maps from a vector for the element to
    the DoF numbering.

    NOTE: if distributed computing is introduced (via MPI), we will need to distinguish
    between `rank` and `global` enumerations.
    """
    connectivity_en = ebc_get_connectivity(
        i=i, E=E, N=N, connectivity=connectivity, EN_offsets=EN_offsets
    )
    # Assumes each node has `U` number of DoFs and DoFs are enumerated following node numbering
    return jnp.vstack(
        [(U[i] * connectivity_en + j).ravel() for j in range(U[i])],
        dtype=jnp.int64,
    ).T.reshape((E[i], N[i] * U[i]))


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
            [b.constitutive_model for b in element_batches]
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
