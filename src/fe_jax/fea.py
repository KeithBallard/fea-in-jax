from .setup import *
from .utils import *
from .solve_cg import cg as cg_w_info
from .sparse_linalg import *

import jax.numpy as jnp
import jax
import jax.experimental.sparse as jsparse
from jax.experimental import mesh_utils

from jaxopt import linear_solve

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any

from flax import struct


@struct.dataclass
class ElementBatch:
    """
    Describes a batch of elements. Passed into solve_bvp()
    """

    fe_type: FiniteElementType = struct.field(pytree_node=False)
    # Number of degrees of freedom per basis function (typically also per a node)
    n_dofs_per_basis: int
    # List of vertex indices for each element (refers to list of vertices passed to solve_bvp(),
    # not internal batch numbering)
    connectivity_en: np.ndarray[Any, np.dtype[np.uint64]]
    constitutive_model: Callable = struct.field(pytree_node=False)
    material_params_eqm: jnp.ndarray
    internal_state_eqi: jnp.ndarray | None = None


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
    E: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)
    # Number of nodes per element for each batch
    # Note: static, not traced
    N: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)
    # Number of degrees of freedom (unknowns) per a node for each batch
    # Note: static, not traced
    U: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)
    # Number of quadrature points per an element for each batch
    # Note: static, not traced
    Q: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)
    # Dimensionality of parametric coordinate system for each batch
    # Note: static, not traced
    P: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)
    # Number of material parameters required for each batch (at a point)
    # Note: static, not traced
    M: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)
    # Number of internal state variables required for each batch (at a point)
    # Note: static, not traced
    I: np.ndarray[Any, np.dtype[np.int64]] = struct.field(pytree_node=False)

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

    # --- Callable functions ---

    # Constitutive model for each batch, length=B
    # Note: static, not traced
    constitutive_models: list[jax.tree_util.Partial] = struct.field(pytree_node=False)

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
    material_params_sizes: np.ndarray[Any, np.dtype[np.int64]] = struct.field(
        pytree_node=False
    )
    # Offset for each batch into `internal_state`, shape=(B+1,)
    internal_state_offsets: jnp.ndarray
    # Size of each batch stored in `internal_state`, shape=(B,)
    internal_state_sizes: np.ndarray[Any, np.dtype[np.int64]] = struct.field(
        pytree_node=False
    )

    # Type of quadrature / basis for each batch, which can be either same quadrature / basis
    # for the entire batch or a different set of quad points / basis for each element.
    # Note: static, not traced
    quadrature_types: list[QuadratureArrayType] = struct.field(pytree_node=False)
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
                    slice_sizes=(self.E[i] * self.Q[i] * self.M[i],),
                ).reshape((self.E[i], self.M[i]))
            case _:  # E
                return jax.lax.dynamic_slice(
                    self.material_params,
                    start_indices=(self.material_params_offsets[i],),
                    slice_sizes=(self.E[i] * self.Q[i] * self.M[i],),
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


def batch_to_collection(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
) -> ElementBatchCollection:
    """
    Converts a list of ElementBatch's to a BatchCollection, which is ameniable to JIT operations.
    """
    E = np.array([b.connectivity_en.shape[0] for b in element_batches])
    N = np.array([b.connectivity_en.shape[1] for b in element_batches])
    U = np.array([b.n_dofs_per_basis for b in element_batches])
    Q = np.array(
        [get_quadrature(fe_type=b.fe_type)[0].shape[0] for b in element_batches]
    )
    M = np.array([b.material_params_eqm.shape[-1] for b in element_batches])
    I = np.array(
        [
            b.internal_state_eqi.shape[-1] if b.internal_state_eqi is not None else 0
            for b in element_batches
        ]
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
        P=np.array([xi_qp.shape[-1] for xi_qp in xi_bqp]),
        I=I,
        # --- Mesh / property / state information ---
        x=jnp.hstack([x_end.ravel() for x_end in x_bend]),
        connectivity=jnp.hstack(
            [b.connectivity_en.ravel() for b in element_batches], dtype=jnp.int64
        ),
        material_params=jnp.hstack(
            [b.material_params_eqm.ravel() for b in element_batches]
        ),
        internal_state=jnp.hstack(
            [
                (
                    b.internal_state_eqi.ravel()
                    if b.internal_state_eqi is not None
                    else jnp.zeros(shape=(E[i], Q[i], I[i]))
                )
                for i, b in enumerate(element_batches)
            ]
        ),
        # --- Quadrature and basis function information ---
        xi=jnp.hstack([xi_qp.ravel() for xi_qp in xi_bqp]),
        weights=jnp.hstack([W_q.ravel() for W_q in W_bq]),
        phi=jnp.hstack([phi_qn.ravel() for phi_qn in phi_bqn]),
        dphi_dxi=jnp.hstack([dphi_dxi_qnp.ravel() for dphi_dxi_qnp in dphi_dxi_bqnp]),
        # --- Callable functions ---
        constitutive_models=[
            jax.tree_util.Partial(b.constitutive_model) for b in element_batches
        ],
        # --- Offsets / sizes into expanded arrays for slicing ---
        EN_offsets=jnp.hstack([jnp.array([0]), jnp.cumsum(E * N)]),
        material_params_types=[
            MaterialPropertyArrayType(len(b.material_params_eqm.shape))
            for b in element_batches
        ],
        material_params_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(
                    jnp.array([b.material_params_eqm.size for b in element_batches])
                ),
            ]
        ),
        material_params_sizes=np.array(
            [b.material_params_eqm.size for b in element_batches]
        ),
        internal_state_offsets=jnp.hstack(
            [
                jnp.array([0]),
                jnp.cumsum(
                    jnp.array(
                        [
                            (
                                b.internal_state_eqi.size
                                if b.internal_state_eqi is not None
                                else 0
                            )
                            for b in element_batches
                        ]
                    )
                ),
            ]
        ),
        internal_state_sizes=np.array(
            [
                b.internal_state_eqi.size if b.internal_state_eqi is not None else 0
                for b in element_batches
            ]
        ),
        quadrature_types=[QuadratureArrayType.Q for b in element_batches],
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


class LinearSolverType(Enum):
    DIRECT_SPARSE_SOLVE_JNP = (0,)
    DIRECT_INVERSE_JNP = 5
    DIRECT_INVERSE_JAXOPT = 6
    CG_JAXOPT = 10
    CG_SCIPY = 11
    CG_SCIPY_W_INFO = 12
    CG_JACOBI_SCIPY = 13
    GMRES_JAXOPT = 20
    GMRES_SCIPY = 21
    BICGSTAB_JAXOPT = 30
    BICGSTAB_SCIPY = 31
    CHOLESKY_JAXOPT = 40
    LU_JAXOPT = 50


@dataclass(eq=True, frozen=True)
class SolverOptions:
    linear_solve_type: LinearSolverType = LinearSolverType.DIRECT_INVERSE_JNP
    linear_relative_tol: float = 1e-14
    linear_absolute_tol: float = 1e-10
    nonlinear_max_iter: int = 1
    nonlinear_relative_tol: float = 1e-12
    nonlinear_absolute_tol: float = 1e-8


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
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the element-level jacobian matrices for a batch of elements without any modification
    of the solution or residual to accomodate Dirichlet constraints.

    TODO document parameters
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
    def residual_kernel(u_t, x_nd, material_params_qm, internal_state_qi):
        u_nd = u_t.reshape(N, D)
        R_nu = element_residual_func(
            u_nd=u_nd,
            x_nd=x_nd,
            dphi_dxi_qnp=dphi_dxi_qnp,
            W_q=W_q,
            material_params_qm=material_params_qm,
            internal_state_qi=internal_state_qi,
            constitutive_model=constitutive_model,
        )[0]
        return R_nu.reshape(N * U)

    J_ett = jax.vmap(jax.jacfwd(residual_kernel, argnums=0))(
        u_et, x_end, material_params_eqm, internal_state_eqi
    )

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
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
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
        material_params_eqm=material_params_eqm,
        internal_state_eqi=internal_state_eqi,
    )
    # debug_print(J_ett)

    return (J_ett, rows, cols)


def calculate_jacobian_wo_dirichlet(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
    precomputed_jacobian_nnz: int,
):

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    J_ett, rows, cols = zip(
        *[
            _calculate_jacobian_coo_terms_batch(
                element_residual_func=element_residual_func,
                constitutive_model=ebc.constitutive_models[i],
                material_params_eqm=ebc.get_material_params(i),
                internal_state_eqi=ebc.get_internal_state(i),
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
    J_ett = jnp.vstack(J_ett)
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
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the element-level jacobian matrices for a batch of elements without any modification
    of the solution or residual to accomodate Dirichlet constraints.

    TODO document parameters
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
    def residual_kernel(u_t, x_nd, material_params_qm, internal_state_qi):
        u_nd = u_t.reshape(N, D)
        R_nu = element_residual_func(
            u_nd=u_nd,
            x_nd=x_nd,
            dphi_dxi_qnp=dphi_dxi_qnp,
            W_q=W_q,
            material_params_qm=material_params_qm,
            internal_state_qi=internal_state_qi,
            constitutive_model=constitutive_model,
        )[0]
        return R_nu.reshape(N * U)

    @jax.vmap
    def vmap_diag_J(u_t, x_nd, material_params_qm, internal_state_qi):
        return jnp.diagonal(
            jax.jacfwd(residual_kernel, argnums=0)(
                u_t, x_nd, material_params_qm, internal_state_qi
            )
        )

    diag_J_et = vmap_diag_J(u_et, x_end, material_params_eqm, internal_state_eqi)

    assert diag_J_et.shape == (
        E,
        N * U,
    ), f"Expected shape {(E, N * U)}, but received {diag_J_et.shape}"

    return diag_J_et


@jax.jit
def _calculate_jacobian_diag_coo_terms_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
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
        material_params_eqm=material_params_eqm,
        internal_state_eqi=internal_state_eqi,
    )
    # debug_print(diag_J_et)

    return (diag_J_et, dof_map)


def calculate_jacobian_diag_wo_dirichlet(
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
                material_params_eqm=ebc.get_material_params(i),
                internal_state_eqi=ebc.get_internal_state(i),
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
def _calculate_residual_wo_dirichlet_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
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
            0,  # material_params_eqm -> material_params_qm
            0,  # internal_state_eqi -> internal_state_qi
            None,  # constitutive_model
        ),
    )

    R_enu, internal_state_eqi = R_vmap(
        u_enu,
        x_end,
        dphi_dxi_qnp,
        W_q,
        material_params_eqm,
        internal_state_eqi,
        constitutive_model,
    )

    return R_enu, internal_state_eqi


def calculate_residual_wo_dirichlet(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
):
    """
    Calculates the residual without any modification of the solution or residual to accomodate
    Dirichlet constraints. Called by calculate_residual.

    TODO document parameters
    """

    # TODO change the pattern to accept donated arrays to hold R_f and new_internal_state_beqi

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    result = [
        _calculate_residual_wo_dirichlet_batch(
            element_residual_func=element_residual_func,
            constitutive_model=ebc.constitutive_models[i],
            material_params_eqm=ebc.get_material_params(i),
            internal_state_eqi=ebc.get_internal_state(i),
            x_end=ebc.get_x(i),
            dphi_dxi_qnp=ebc.get_dphi_dxi(i),
            W_q=ebc.get_weights(i),
            assembly_map=assembly_map_b[i],
            u_f=u_f,
        )
        for i in range(ebc.B)
    ]  # for each item, 0: R_end, 1: internal_state_eqi

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
        R_enu, internal_state_eqi = _calculate_residual_wo_dirichlet_batch(
            element_residual_func=element_residual_func,
            constitutive_model=constitutive_model_b[i],
            material_params_eqm=material_params_beqm[i],
            internal_state_eqi=internal_state_beqi[i],
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


def calculate_residual_w_dirichlet(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    u_f: jnp.ndarray,
    dirichlet_values_g: jnp.ndarray,
    dirichlet_mask_g: jnp.ndarray,
):
    """
    Compute the residual vector given the current solution and state information.
    TODO document better

    Parameters
    ----------
    u_0_g         : initial guess for the solution in the current linear solve (nonlinear constitutive
                    models will be linearized about this point), dense 1d-array of length V * D
    u_f           : current solution within the linear solve, dense 1d-array of length V * D

    Returns
    -------
    R_e  : dense 1d-array with shape (N_gn * N_u)
    """

    # Note: this is neccessary to ensure the Jacobian is symmetric. Without this,
    # the autodiff would result in 0's on rows (except on the diagonal) for entries
    # corresponding to Dirichlet BC's, but the columns would be non-zero.
    u_f_w_dirichlet = jnp.multiply(1.0 - dirichlet_mask_g, u_f) + jnp.multiply(
        dirichlet_mask_g, dirichlet_values_g
    )

    R_f, new_internal_state_beqi = calculate_residual_wo_dirichlet(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f_w_dirichlet,
    )

    # Zero out terms corresponding to Dirichlet BCs and add (solution - what it should be) for those constrained DoFs.
    # This will ensure there will be a 1 on the diagonal of the Jacobian and also return the right residual.
    R_f = jnp.multiply(1.0 - dirichlet_mask_g, R_f) + jnp.multiply(
        dirichlet_mask_g, u_f - dirichlet_values_g
    )

    return R_f, new_internal_state_beqi


def solve_nonlinear_step(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    jacobian_nnz: int,
    u_0_g: jnp.ndarray,
    dirichlet_values_g: jnp.ndarray,
    dirichlet_mask_g: jnp.ndarray,
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
    solver_options: SolverOptions,
):
    """
    Solve the linearized system of equations emerging from the governing equations.
    This can be used within an outer loop to solve linear PDEs across time steps with different
    boundary conditions or to solve a nonlinear problem (via Newton's method for example).

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    constitutive_model_b  : constitutive model relating stress-strain for each element batch
    material_params_beqm  : material parameters for each element batch, [ndarray[float, (E, Q, M)]]
    x_bend                : nodal coordinates in each element for each element batch, [ndarray[float, (E, N, D)]]
    dphi_dxi_bqnp         : derivative of basis function in parameteric coordinate system evaluated
                            at each quadrature point for each element batch, [ndarray[float, (E, Q, M)]]
    W_bq                  : quadrature weights for each element match, [ndarray[float, (Q,)]]
    assembly_map_b        : at map for which the matmult provides assembly for each element batch,
                            [sparse[float, (V, E*N)]]
    u_0_g                 : initial solution, ndarray[float, (V * D)]
    dirichlet_values_g    : value specified for Dirichlet boundary conditions, ndarray[float, (V * D)]
    dirichlet_mask_g      : mask that is 1 for DoFs corresponding to Dirichlet boundary conditions and 0
                            otherwise, ndarray[float, (V * D)]
    dirichlet_dofs        : list of DoFs for Dirichlet boundary conditions, ndarray[int, (# Dirichlet BCs,)]
    dirichlet_values      : values of Dirichlet boundary conditions, ndarray[float, (# Dirichlet BCs,)]
    linear_solver_type    : type of linear solver to use
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
    residual_isv_func_w_dirichlet = lambda u_f: calculate_residual_w_dirichlet(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
    )

    # Function that produces R(u)
    residual_func_w_dirichlet = lambda u_f: residual_isv_func_w_dirichlet(u_f=u_f)[0]

    # Function that produces R(u) without Dirichlet BCs applied
    residual_func_wo_dirichlet = lambda u_f: calculate_residual_wo_dirichlet(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f,
    )[0]

    # Function that produces J(u) without Dirichlet BCs applied
    jacobian_func_wo_dirichlet = lambda u_f: calculate_jacobian_wo_dirichlet(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f,
        precomputed_jacobian_nnz=jacobian_nnz,
    )

    # Function that produces diag(J(u)) without Dirichlet BCs applied
    jacobian_diag_func_wo_dirichlet = lambda u_f: calculate_jacobian_diag_wo_dirichlet(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f,
    )

    R_f, new_internal_state_beqi = residual_isv_func_w_dirichlet(u_f=u_0_g)
    initial_R_f_norm = jnp.linalg.norm(R_f)

    # Note: will be specialized for u_f later in while_body
    jacobian_vector_product_detail = lambda u_f, z: jax.jvp(
        residual_func_w_dirichlet,
        (u_f,),
        (z,),
    )[1]

    # NOTE Used to debug if the Jacobian via autodiff matches the Jacobian via assembly
    """
    jacobian = jax.jacfwd(residual_func_wo_dirichlet)(u_0_g)
    cp.savetxt('A_jacfwd.csv', cp.asarray(jacobian))

    jacobian_w_bc = jax.jacfwd(residual_func_w_dirichlet)(u_0_g)
    cp.savetxt('A_jacfwd_w_bc.csv', cp.asarray(jacobian_w_bc))

    J_sparse_ff = jacobian_func_wo_dirichlet(u_0_g)
    cp.savetxt('A_sparse_ff.csv', cp.asarray(J_sparse_ff.todense()))

    lhs_matrix, rhs_vector = apply_dirichlet_bcs(
        J_sparse_ff,
        -R_f,
        dirichlet_dofs,
        dirichlet_values - u_0_g[dirichlet_dofs],
    )
    cp.savetxt('A_sparse_ff_w_bc.csv', cp.asarray(lhs_matrix.todense()))

    assert jnp.isclose(J_sparse_ff.todense(), jacobian).all()
    assert jnp.isclose(lhs_matrix.todense(), jacobian_w_bc).all()
    """

    def while_cond(args) -> bool:
        nl_iteration, u_f, R_f, new_internal_state_beqi = args
        absolute_error = jnp.linalg.norm(R_f)
        relative_error = absolute_error / initial_R_f_norm
        jax.debug.print(
            "Iteration {z} rel error {x}, abs error {y}",
            x=relative_error,
            y=absolute_error,
            z=nl_iteration,
        )
        return (
            (nl_iteration < solver_options.nonlinear_max_iter)
            & (relative_error > solver_options.nonlinear_relative_tol)
            & (absolute_error > solver_options.nonlinear_absolute_tol)
        )

    def while_body(args) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        nl_iteration, u_f, R_f, new_internal_state_beqi = args
        # jax.debug.print("u_f = {x}", x=u_f)

        # Note: unclear which is most performant variant of this.
        # Function that produces J(u) * z with Dirichlet constraints
        # Note: this linearizes the Jacobian about u_0
        # jacobian_vector_product = lambda z: jax.jvp(
        #    residual_func_w_dirichlet,
        #    (u_f,),
        #    (z,),
        # )[1]
        # jacobian_vector_product_inner = jax.tree_util.Partial(residual_func_w_dirichlet, (u_f,))
        # jacobian_vector_product = lambda z: jacobian_vector_product_detail(u_f, z)
        jacobian_vector_product = jax.tree_util.Partial(
            jacobian_vector_product_detail, u_f
        )

        # Solve the boundary value problem
        info = None
        match solver_options.linear_solve_type:

            case LinearSolverType.DIRECT_SPARSE_SOLVE_JNP:
                # NOTE Forms the sparse Jacobian in memory
                J_sparse_ff = jacobian_func_wo_dirichlet(u_0_g)
                J_sparse_ff = apply_dirichlet_bcs_lhs(
                    J_sparse_ff,
                    dirichlet_dofs,
                )
                delta_u = spsolve(J_sparse_ff, -R_f)

            case LinearSolverType.DIRECT_INVERSE_JNP:
                # NOTE Forms the dense Jacobian matrix in memory
                # NOTE jacfwd of residual_func_w_dirichlet will automatically include in-place
                #      elimination of Dirichlet BCs.
                jacobian = jax.jacfwd(residual_func_w_dirichlet)(u_0_g)
                delta_u = jnp.array(jnp.dot(jnp.linalg.inv(jacobian), -R_f))

            case LinearSolverType.DIRECT_INVERSE_JAXOPT:
                delta_u = linear_solve.solve_inv(matvec=jacobian_vector_product, b=-R_f)

            case LinearSolverType.CG_SCIPY:
                delta_u, _ = jax.scipy.sparse.linalg.cg(
                    A=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.CG_SCIPY_W_INFO:
                delta_u, info = cg_w_info(
                    A=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.CG_JACOBI_SCIPY:
                diag_J_f = jacobian_diag_func_wo_dirichlet(u_0_g)
                M_inv = 1.0 / diag_J_f
                u, _ = jax.scipy.sparse.linalg.cg(
                    A=jacobian_vector_product,
                    M=M_inv,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.CG_JAXOPT:
                delta_u = linear_solve.solve_cg(
                    matvec=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.GMRES_SCIPY:
                delta_u, _ = jax.scipy.sparse.linalg.gmres(
                    A=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.GMRES_JAXOPT:
                delta_u = linear_solve.solve_gmres(
                    matvec=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.BICGSTAB_SCIPY:
                delta_u, _ = jax.scipy.sparse.linalg.bicgstab(
                    A=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.BICGSTAB_JAXOPT:
                delta_u = linear_solve.solve_bicgstab(
                    matvec=jacobian_vector_product,
                    b=-R_f,
                    tol=solver_options.linear_relative_tol,
                    atol=solver_options.linear_absolute_tol,
                )

            case LinearSolverType.CHOLESKY_JAXOPT:
                delta_u = linear_solve.solve_cholesky(
                    matvec=jacobian_vector_product, b=-R_f
                )

            case LinearSolverType.LU_JAXOPT:
                delta_u = linear_solve.solve_lu(matvec=jacobian_vector_product, b=-R_f)

            case _:
                raise Exception(
                    f"Linear solver type {solver_options.linear_solve_type} is not implemented"
                )

        # Note: consider implementing spai preconditioner
        # https://tbetcke.github.io/hpc_lecture_notes/it_solvers4.html

        # jax.scipy solvers will not arrive at the right values for the constraints for any size of
        # problem but even the jaxopt solvers will only get close for large enough problems.
        # Consequently, overwrite the values directly to ensure the BCs are right, even though the
        # residual may increase.
        delta_u = jnp.multiply(1.0 - dirichlet_mask_g, delta_u) + jnp.multiply(
            dirichlet_mask_g, dirichlet_values_g - u_f
        )
        # jax.debug.print("delta u = {x}", x=delta_u)

        u_f = u_f + delta_u
        R_f = residual_isv_func_w_dirichlet(u_f=u_f)[0]

        return (nl_iteration + 1, u_f, R_f, new_internal_state_beqi)

    _, u_f, R_f, new_internal_state_beqi = jax.lax.while_loop(
        cond_fun=while_cond,
        body_fun=while_body,
        init_val=(0, u_0_g, R_f, new_internal_state_beqi),
    )

    absolute_error = jnp.linalg.norm(R_f)
    relative_error = absolute_error / initial_R_f_norm
    return (u_f, new_internal_state_beqi, R_f, relative_error, None)


def solve_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: Callable,
    u_0_g: jnp.ndarray | None,
    dirichlet_bcs: np.ndarray[Any, np.dtype[np.uint64]],
    dirichlet_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
    solver_options: SolverOptions = SolverOptions(),
    plot_convergence: bool = False,
    profile_memory: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, list[ElementBatch]]:
    """
    Solve a boundary value problem for static linear elasticity.

    Parameters
    ----------
    vertices_vd          : vertices needed for all cells on the rank, ndarray[float, (V, D)]
    element_batches      : batch of elements for this rank
    element_residual_func: residual function emerging from weak form of governing equations
    dirichlet_bcs        : Dirichlet boundary conditions, ndarray[int, (# of constrained DoFs, 2)]
                           with each row having the structure (vertex index, component of solution)
    dirichlet_values     : value specified for Dirichlet boundary conditions, ndarray[float, (# of constrained DoFs,)]
    material_params_beqm : material parameters for each element batch, [ndarray[float, (E, Q, M)]]
    linear_solver_type   : type of linear solver to use whether one is needed for a global solution
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

    B = len(element_batches)
    V = vertices_vd.shape[0]
    D = vertices_vd.shape[1]

    # Validate input
    assert D <= 3
    assert dirichlet_bcs.shape[0] <= D * V
    assert dirichlet_bcs.shape[1] == 2
    assert dirichlet_values.shape[0] == dirichlet_bcs.shape[0]
    for b in element_batches:
        assert b.connectivity_en.shape[0] == b.material_params_eqm.shape[0]
        assert b.connectivity_en.shape[1] <= V

    # Wrap the provided callable to be compatible with jit
    element_residual_func = jax.tree_util.Partial(element_residual_func)

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Convert element batch information into something ameniable to JAX transforms like JIT
    ebc = batch_to_collection(vertices_vd=vertices_vd, element_batches=element_batches)
    # print(ebc)

    assert (
        ebc.U == ebc.U[0]
    ).all(), """The number of DoFs per a point (U) must be the same across all batches.
    To relax this constrain much of the infrastructure code in fea.py would have to be adapted to
    support varying number of DoFs per a batch.
    """

    # If an initial guess was not provided, then use zeros
    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(V * ebc.U[0],))
    else:
        assert u_0_g.shape == (V * ebc.U[0],)

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Compute the anticipated number of non-zeros for the assembled Jacobian, which
    # is only needed for solvers that actually form the Jacobian in memory.
    # NOTE: we need a concrete value to specialize for JIT of other functions
    jacobian_nnz = int(_calculate_jacobian_unique_nnz(n_vertices=V, ebc=ebc))

    # TODO consider JIT'ing this group of lines pending profiling
    # A list of degrees of freedom for the Dirichlet boundary conditions
    dirichlet_dofs = jnp.array(D * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1])
    # print('dirichlet_dofs: ', dirichlet_dofs)
    # Global unraveled
    dirichlet_values_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(dirichlet_values)
    # A global vector with 0's where values are not boundary conditions,
    # and 1's corresponding to Dirichlet BCs.
    dirichlet_mask_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(1.0)

    # TODO move this to ElementBatchCollection as a method if needed
    """
    # Check if the input batches of arrays are all same shape for each batch
    is_batch_homogeneous = lambda batch_arr: all(
        map(lambda arr: arr.shape == batch_arr[0].shape, batch_arr)
    )
    is_x_homogeneous = is_batch_homogeneous(x_bend)
    is_dphi_dxi_homogeneous = is_batch_homogeneous(dphi_dxi_bqnp)
    is_W_homogeneous = is_batch_homogeneous(W_bq)
    # Check if the element batches of arrays are all same shape / type for each batch
    is_fe_type_homogeneous = all(
        map(lambda b: b.fe_type == element_batches[0].fe_type, element_batches)
    )
    is_conn_homogeneous = all(
        map(
            lambda b: b.connectivity_en.shape
            == element_batches[0].connectivity_en.shape,
            element_batches,
        )
    )
    is_mat_params_homogeneous = all(
        map(
            lambda b: b.material_params_eqm.shape
            == element_batches[0].material_params_eqm.shape,
            element_batches,
        )
    )
    # If all of the checks are true, then we an JIT compile the functions
    is_homogeneous = (
        is_x_homogeneous
        and is_dphi_dxi_homogeneous
        and is_W_homogeneous
        and is_fe_type_homogeneous
        and is_conn_homogeneous
        and is_mat_params_homogeneous
    )
    """

    inner_solve = solve_nonlinear_step
    is_homogeneous = False
    if is_homogeneous:
        print("Batches are homogeneous, using JIT compilation for solve_linear_step")
        inner_solve = jax.jit(
            solve_nonlinear_step,
            donate_argnames="internal_state_beqi",
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
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=jnp.array(dirichlet_values),
        solver_options=solver_options,
    )

    # Update internal state variables for the element batches
    # TODO need to update
    # for i, b in enumerate(element_batches):
    #    b.internal_state_eqi = internal_state_beqi[i]

    # capture memory usage after and analyze
    if profile_memory:
        u.block_until_ready()
        stop_memory_profile("solve_linear_step")

    print(f"solver relative error: {relative_error}")
    if info is not None:
        print(f"solver # of iterations: {info['iterations']}")

        if plot_convergence:

            import matplotlib.pyplot as plt

            x_iter = jnp.linspace(
                0, info["iterations"], info["iterations"] + 1, dtype=jnp.int32
            )
            y_r_norm = info["residual_norm_history"][0 : info["iterations"] + 1]

            plt.plot(x_iter, y_r_norm)
            plt.title(
                f"Residual History During Iteration Using {solver_options.linear_solve_type}"
            )
            plt.xlabel("iteration")
            plt.ylabel("|R|")
            plt.yscale("log")
            plt.show()

    return (u, residual, element_batches)
