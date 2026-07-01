from .setup import *
from .utils import *
from .solve_cg import cg as cg_w_info

import jax.numpy as jnp
import jax
import jax.experimental.sparse as jsparse
from jax.experimental import mesh_utils

from jaxopt import linear_solve

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any,NamedTuple


@dataclass
class ElementBatch:
    """
    Describes a batch of elements. Passed into solve_bvp()
    """

    fe_type: FiniteElementType
    # list of vertex indices for each element (refers to list of vertices passed to solve_bvp(), not internal batch numbering)
    connectivity_en: np.ndarray[Any, np.dtype[np.uint64]]
    constitutive_model: Callable
    material_params_eqm: jnp.ndarray
    internal_state_eqi: jnp.ndarray


class LinearSolverType(Enum):
    DIRECT_INVERSE_JNP = 0
    DIRECT_INVERSE_JAXOPT = 0
    CG_JAXOPT = 10
    CG_SCIPY = 11
    CG_SCIPY_W_INFO = 12
    # CG_JACOBI_SCIPY = 13
    GMRES_JAXOPT = 20
    GMRES_SCIPY = 21
    BICGSTAB_JAXOPT = 30
    BICGSTAB_SCIPY = 31
    CHOLESKY_JAXOPT = 40
    LU_JAXOPT = 50


@jax.jit
def _calculate_residual_batch_element_kernel(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    u_end: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
):
    """
    Calculates the element-level residual vectors for a batch of elements without any modification
    of the solution or residual to accomodate Dirichlet constraints. Called by calculate_residual.

    TODO document parameters
    """

    @jax.vmap
    def residual_kernel(u_nd, x_nd, material_params_qm, internal_state_qi):
        return element_residual_func(
            constitutive_model=constitutive_model,
            u_nd=u_nd,
            x_nd=x_nd,
            dphi_dxi_qnp=dphi_dxi_qnp,
            W_q=W_q,
            material_params_qm=material_params_qm,
            internal_state_qi=internal_state_qi,
        )

    R_end, internal_state_eqi = residual_kernel(
        u_end, x_end, material_params_eqm, internal_state_eqi
    )

    return R_end, internal_state_eqi


@jax.jit
def __calculate_residual_wo_dirichlet_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params_eqm: jnp.ndarray,
    internal_state_eqi: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    assembly_map: jsparse.BCSR,
    u_g: jnp.ndarray,
):
    # Extract shape constants needed for args
    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]

    u_end = transform_global_unraveled_to_element_node(
        assembly_map, u_g, N_ge=E, N_n=N, N_u=D
    )

    R_end, internal_state_eqi = _calculate_residual_batch_element_kernel(
        element_residual_func=element_residual_func,
        constitutive_model=constitutive_model,
        u_end=u_end,
        x_end=x_end,
        dphi_dxi_qnp=dphi_dxi_qnp,
        W_q=W_q,
        material_params_eqm=material_params_eqm,
        internal_state_eqi=internal_state_eqi,
    )

    return R_end, internal_state_eqi


def calculate_residual_wo_dirichlet(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model_b: list[jax.tree_util.Partial],
    material_params_beqm: list[jnp.ndarray],
    internal_state_beqi: list[jnp.ndarray],
    x_bend: list[jnp.ndarray],
    dphi_dxi_bqnp: list[jnp.ndarray],
    W_bq: list[jnp.ndarray],
    assembly_map_b: list[jsparse.BCSR],
    u_g: jnp.ndarray,
):
    """
    Calculates the residual without any modification of the solution or residual to accomodate
    Dirichlet constraints. Called by calculate_residual.

    TODO document parameters
    """

    B = len(x_bend)

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    result = [
        __calculate_residual_wo_dirichlet_batch(
            element_residual_func=element_residual_func,
            constitutive_model=constitutive_model_b[i],
            material_params_eqm=material_params_beqm[i],
            internal_state_eqi=internal_state_beqi[i],
            x_end=x_bend[i],
            dphi_dxi_qnp=dphi_dxi_bqnp[i],
            W_q=W_bq[i],
            assembly_map=assembly_map_b[i],
            u_g=u_g,
        )
        for i in range(B)
    ]  # for each item, 0: R_end, 1: internal_state_eqi

    R_g = jnp.zeros_like(u_g)
    for i in range(B):
        R_g += transform_element_node_to_global_unraveled_sum(
            assembly_map=assembly_map_b[i], v_en=result[i][0]
        )

    new_internal_state_beqi = [result[i][1] for i in range(B)]
    # TODO split this out into a separate call

    return R_g, new_internal_state_beqi


def calculate_residual_w_dirichlet(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model_b: list[jax.tree_util.Partial],
    material_params_beqm: list[jnp.ndarray],
    internal_state_beqi: list[jnp.ndarray],
    x_bend: list[jnp.ndarray],
    dphi_dxi_bqnp: list[jnp.ndarray],
    W_bq: list[jnp.ndarray],
    assembly_map_b: list[jsparse.BCSR],
    u_g: jnp.ndarray,
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
    u_g           : current solution within the linear solve, dense 1d-array of length V * D

    Returns
    -------
    R_e  : dense 1d-array with shape (N_gn * N_u)
    """

    # Note: this is neccessary to ensure the Jacobian is symmetric. Without this,
    # the autodiff would result in 0's on rows (except on the diagonal) for entries
    # corresponding to Dirichlet BC's, but the columns would be non-zero.
    u_g_w_dirichlet = jnp.multiply(1.0 - dirichlet_mask_g, u_g) + jnp.multiply(
        dirichlet_mask_g, dirichlet_values_g
    )

    R_g, new_internal_state_beqi = calculate_residual_wo_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_g_w_dirichlet,
    )

    # Zero out terms corresponding to Dirichlet BCs and add (solution - what it should be) for those constrained DoFs.
    # This will ensure there will be a 1 on the diagonal of the Jacobian and also return the right residual.
    R_g = jnp.multiply(1.0 - dirichlet_mask_g, R_g) + jnp.multiply(
        dirichlet_mask_g, u_g - dirichlet_values_g
    )

    return R_g, new_internal_state_beqi


"""
# @partial(jax.jit, static_argnames=["N_ge", "N_n", "N_u"])
def calculate_jacobian_diagonal(
    u_gn: jnp.ndarray,
    x_esd: jnp.ndarray,
    dphi_dxi_bqsp: jnp.ndarray,
    W_q: jnp.ndarray,
    mat_params_eq: jnp.ndarray,
    assembly_map: jsparse.BCSR,
    N_ge: int,
    N_n: int,
    N_u: int,
):
    # Function that produces element-level residuals
    residual_element_batch = (
        lambda u_esd, R_gn: _calculate_residual_batch_element_kernel(
            u_esd=u_esd,
            x_esd=x_esd,
            dphi_dxi_bqsp=dphi_dxi_bqsp,
            W_q=W_q,
            mat_params_eq=mat_params_eq,
        )
    )

    u_esd = transform_global_unraveled_to_element_node(
        assembly_map, u_gn, N_ge=N_ge, N_n=N_n, N_u=N_u
    )

    jax.lax.fori_loop(lower=0, upper=u_esd.shape[0], body_fun=None, init_val=None)
    jacobian_element_batch = jax.jacfwd(residual_element_batch)(u_esd)
    print("J_batch", jacobian_element_batch.shape, jacobian_element_batch)
    jacobian_element_batch_diag = jnp.diagonal(
        a=jacobian_element_batch, axis1=2, axis2=3
    )

    print("jacobian_element_batch_diag", jacobian_element_batch_diag.shape)

    J_diag = transform_element_node_to_global_unraveled_sum(
        assembly_map=assembly_map, v_en=jacobian_element_batch_diag
    )

    return J_diag
"""


def solve_linear_step(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model_b: list[jax.tree_util.Partial],
    material_params_beqm: list[jnp.ndarray],
    internal_state_beqi: list[jnp.ndarray],
    x_bend: list[jnp.ndarray],
    dphi_dxi_bqnp: list[jnp.ndarray],
    W_bq: list[jnp.ndarray],
    assembly_map_b: list[jsparse.BCSR],
    u_0_g: jnp.ndarray,
    dirichlet_values_g: jnp.ndarray,
    dirichlet_mask_g: jnp.ndarray,
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
    linear_solver_type: LinearSolverType,
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
    """
    B = len(x_bend)
    print(f"# of batches: {B}")
    for i in range(B):
        E = x_bend[i].shape[0]
        N = x_bend[i].shape[1]
        D = x_bend[i].shape[2]
        Q = dphi_dxi_bqnp[i].shape[0]
        P = dphi_dxi_bqnp[i].shape[2]
        M = material_params_beqm[i].shape[2]
        print(
            f"For batch {i}:\n\t",
            f"Number of elements : {E}\n\t",
            f"Number of nodes / element : {N}\n\t",
            f"Global dimensionality : {D}\n\t",
            f"Number of quadrature points : {Q}\n\t",
            f"Parametric dimensionality: {P}\n\t",
            f"Number of material parameters per quad point: {M}",
        )
    """

    # Function that produces R(u)
    residual_func_w_dirichlet = lambda u_g: calculate_residual_w_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_g,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
    )[0]

    # Function that produces R(u) without Dirichlet BCs applied
    residual_func_wo_dirichlet = lambda u_g: calculate_residual_wo_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_g,
    )[0]

    # Adjust right-hand side vector for Dirichlet constraints
    jacobian_product_wo_dirichlet = lambda z: jax.jvp(
        residual_func_wo_dirichlet,
        (u_0_g,),
        (z,),
    )[1]
    b = -jacobian_product_wo_dirichlet(dirichlet_values_g)
    b = b.at[dirichlet_dofs].set(dirichlet_values)

    # Function that produces J(u) * z with Dirichlet constraints
    # Note: this linearizes the Jacobian about u_0
    jacobian_vector_product = lambda z: jax.jvp(
        residual_func_w_dirichlet,
        (u_0_g,),
        (z,),
    )[1]

    """
    # Function that produces diag(J(u)) without Dirichlet constraints
    jacobian_diagonal_func = lambda u_gn: calculate_jacobian_diagonal(
        u_gn=u_gn,
        x_esd=x_esd,
        dphi_dxi_bqsp=dphi_dxi_bqsp,
        W_q=W_q,
        mat_params_eq=mat_params_eq,
        assembly_map=assembly_map,
        N_ge=N_ge,
        N_n=N_n,
        N_u=N_u,
    )
    """

    # Solve the boundary value problem
    info = None
    match linear_solver_type:
        case LinearSolverType.DIRECT_INVERSE_JNP:
            # Calculate the Jacobian matrix in-memory
            jacobian = jax.jacfwd(residual_func_w_dirichlet)(u_0_g)
            u = jnp.array(jnp.dot(jnp.linalg.inv(jacobian), b))

        case LinearSolverType.DIRECT_INVERSE_JAXOPT:
            u = linear_solve.solve_inv(matvec=jacobian_vector_product, b=b)

        case LinearSolverType.CG_SCIPY:
            u, _ = jax.scipy.sparse.linalg.cg(A=jacobian_vector_product, b=b)

        case LinearSolverType.CG_SCIPY_W_INFO:
            u, info = cg_w_info(A=jacobian_vector_product, b=b, tol=1e-9)

        # case LinearSolverType.CG_JACOBI_SCIPY:
        # M_inv = 1.0 / jacobian_diagonal_func(u_0_g)
        # u, _ = jax.scipy.sparse.linalg.cg(A=jacobian_vector_product, M=M_inv, b=b)

        case LinearSolverType.CG_JAXOPT:
            u = linear_solve.solve_cg(matvec=jacobian_vector_product, b=b, tol=1e-8)

        case LinearSolverType.GMRES_SCIPY:
            u, _ = jax.scipy.sparse.linalg.gmres(A=jacobian_vector_product, b=b)

        case LinearSolverType.GMRES_JAXOPT:
            u = linear_solve.solve_gmres(matvec=jacobian_vector_product, b=b, tol=1e-8)

        case LinearSolverType.BICGSTAB_SCIPY:
            u, _ = jax.scipy.sparse.linalg.bicgstab(
                A=jacobian_vector_product, b=b, tol=1e-8
            )

        case LinearSolverType.BICGSTAB_JAXOPT:
            u = linear_solve.solve_bicgstab(
                matvec=jacobian_vector_product, b=b, tol=1e-8
            )

        case LinearSolverType.CHOLESKY_JAXOPT:
            u = linear_solve.solve_cholesky(matvec=jacobian_vector_product, b=b)

        case LinearSolverType.LU_JAXOPT:
            u = linear_solve.solve_lu(matvec=jacobian_vector_product, b=b)

        case _:
            raise Exception(
                f"Linear solver type {linear_solver_type} is not implemented"
            )

    # Note: consider implementing spai preconditioner
    # https://tbetcke.github.io/hpc_lecture_notes/it_solvers4.html

    # jax.scipy solvers will not arrive at the right values for the constraints for any size of
    # problem but even the jaxopt solvers will only get close for large enough problems.
    # Consequently, overwrite the values directly to ensure the BCs are right, even though the
    # residual may increase.
    u = u.at[dirichlet_dofs].set(dirichlet_values)

    R_final, new_internal_state_beqi = calculate_residual_w_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
    )
    relative_error = jnp.linalg.norm(R_final) / jnp.linalg.norm(b)

    return (u, new_internal_state_beqi, R_final, relative_error, info)



def solve_nonlinear_step(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model_b: list[jax.tree_util.Partial],
    material_params_beqm: list[jnp.ndarray],
    internal_state_beqi: list[jnp.ndarray],
    x_bend: list[jnp.ndarray],
    dphi_dxi_bqnp: list[jnp.ndarray],
    W_bq: list[jnp.ndarray],
    assembly_map_b: list[jsparse.BCSR],
    u_0_g: jnp.ndarray,
    dirichlet_values_g: jnp.ndarray,
    dirichlet_mask_g: jnp.ndarray,
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
    linear_solver_type: LinearSolverType,
    tol=1e-8, max_iter=20
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
    """
    B = len(x_bend)
    print(f"# of batches: {B}")
    for i in range(B):
        E = x_bend[i].shape[0]
        N = x_bend[i].shape[1]
        D = x_bend[i].shape[2]
        Q = dphi_dxi_bqnp[i].shape[0]
        P = dphi_dxi_bqnp[i].shape[2]
        M = material_params_beqm[i].shape[2]
        print(
            f"For batch {i}:\n\t",
            f"Number of elements : {E}\n\t",
            f"Number of nodes / element : {N}\n\t",
            f"Global dimensionality : {D}\n\t",
            f"Number of quadrature points : {Q}\n\t",
            f"Parametric dimensionality: {P}\n\t",
            f"Number of material parameters per quad point: {M}",
        )
    """

    # Function that produces R(u)
    residual_func_w_dirichlet = lambda u_g: calculate_residual_w_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_g,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
    )[0]

    # Function that produces R(u) without Dirichlet BCs applied
    residual_func_wo_dirichlet = lambda u_g: calculate_residual_wo_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_g,
    )[0]

    # Adjust right-hand side vector for Dirichlet constraints
    jacobian_product_wo_dirichlet = lambda z: jax.jvp(
        residual_func_wo_dirichlet,
        (u_0_g,),
        (z,),
    )[1]

    # Calculate du
    du_g = jnp.zeros_like(u_0_g)
    du_g = du_g.at[dirichlet_dofs].set(dirichlet_values - u_0_g[dirichlet_dofs])

    b = -jacobian_product_wo_dirichlet(du_g)
    # Dirchelete is the du not the u
    b = b.at[dirichlet_dofs].set(du_g[dirichlet_dofs])

    # --- Define Jacobian-vector product (J @ z) ---
    jacobian_vector_product = lambda z: jax.jvp(
        residual_func_w_dirichlet,
        (u,),
        (z,),
    )[1]

    # ---  Solve the linearized Newton step: J @ delta_u = -R ---
    info = None
    match linear_solver_type:
        case LinearSolverType.DIRECT_INVERSE_JNP:
            # Calculate the Jacobian matrix in-memory
            jacobian = jax.jacfwd(residual_func_w_dirichlet)(u_0_g)
            delta_u = jnp.dot(jnp.linalg.inv(jacobian), b)
        # case LinearSolverType.CG_SCIPY_W_INFO:
        #     delta_u, info = cg_w_info(A=jacobian_vector_product, b=b, tol=1e-9)

    u = u_0_g + delta_u
    u = u.at[dirichlet_dofs].set(dirichlet_values)

    # info = None
    # u = u_0_g

    R_final, new_internal_state_beqi = calculate_residual_w_dirichlet(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_g=u_0_g,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
    )
    R_final = R_final.at[dirichlet_dofs].set(0)

    relative_error = jnp.linalg.norm(R_final) / jnp.linalg.norm(b)
    Res_norm = jnp.linalg.norm(R_final)
    rel_tol = tol * Res_norm

    iter = 0
    while Res_norm > tol and iter < max_iter:
    # while Res_norm > rel_tol and iter < max_iter:
        b = -residual_func_w_dirichlet(u)
        # --- (3) Solve the linearized Newton step: J @ delta_u = -R ---
        info = None
        match linear_solver_type:
            case LinearSolverType.DIRECT_INVERSE_JNP:
                # Calculate the Jacobian matrix in-memory
                jacobian = jax.jacfwd(residual_func_w_dirichlet)(u)
                delta_u = jnp.dot(jnp.linalg.inv(jacobian), b)
            case LinearSolverType.CG_SCIPY_W_INFO:
                delta_u, info = cg_w_info(A=jacobian_vector_product, b=b, tol=1e-9)

        u = u + delta_u
        u = u.at[dirichlet_dofs].set(dirichlet_values)

        R_final, new_internal_state_beqi = calculate_residual_w_dirichlet(
            element_residual_func=element_residual_func,
            constitutive_model_b=constitutive_model_b,
            material_params_beqm=material_params_beqm,
            internal_state_beqi=internal_state_beqi,
            x_bend=x_bend,
            dphi_dxi_bqnp=dphi_dxi_bqnp,
            W_bq=W_bq,
            assembly_map_b=assembly_map_b,
            u_g=u,
            dirichlet_values_g=dirichlet_values_g,
            dirichlet_mask_g=dirichlet_mask_g,
        )
        Res_norm = jnp.linalg.norm(R_final) * tol

        relative_error = jnp.linalg.norm(R_final) / jnp.linalg.norm(b)
        iter += 1

    return (u, new_internal_state_beqi, R_final, relative_error, info)



def solve_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.float32]],
    element_batches: list[ElementBatch],
    element_residual_func: Callable,
    u_0_g: jnp.ndarray | None,
    dirichlet_bcs: np.ndarray[Any, np.dtype[np.uint64]],
    dirichlet_values: np.ndarray[Any, np.dtype[np.float32]],
    linear_solver_type: LinearSolverType = LinearSolverType.DIRECT_INVERSE_JNP,
    plot_convergence: bool = False,
    profile_memory: bool = False,
):
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
    profile_memory     : indicates if GPU memory usage should be profiled, which will create *.prof
                         files in the current directory

    Returns
    -------
    u  : solution (displacement), ndarray[float, (V * D)]
    R  : residual vector evaluated at the solution, ndarray[float, (V * D)]
    """

    B = len(element_batches)
    V = vertices_vd.shape[0]
    D = vertices_vd.shape[1]

    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(V * D,))

    assert D <= 3
    assert u_0_g.shape == (V * D,)
    assert dirichlet_bcs.shape[0] <= D * V
    assert dirichlet_bcs.shape[1] == 2
    assert dirichlet_values.shape[0] == dirichlet_bcs.shape[0]
    for b in element_batches:
        assert b.connectivity_en.shape[0] == b.material_params_eqm.shape[0]
        assert b.connectivity_en.shape[1] <= V

    # For each batch find the list of vertices that are unique to form a local
    # numbering for each batch.
    # batch_to_global_map = [np.unique(b.connectivity_en) for b in element_batches]

    # Setup inputs
    # Wrap the provided callables to be compatible with jit
    element_residual_func = jax.tree_util.Partial(element_residual_func)
    constitutive_model_b = [
        jax.tree_util.Partial(b.constitutive_model) for b in element_batches
    ]
    xi_bqp, W_bq = zip(*[get_quadrature(fe_type=b.fe_type) for b in element_batches])
    phi_bqn, dphi_dxi_bqnp = zip(
        *[
            eval_basis_and_derivatives(fe_type=b.fe_type, xi_qp=xi_bqp[i])
            for i, b in enumerate(element_batches)
        ]
    )

    material_params_beqm = [b.material_params_eqm for b in element_batches]
    internal_state_beqi = [b.internal_state_eqi for b in element_batches]
    x_bend = [
        mesh_to_jax(vertices=vertices_vd, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # TODO JIT this group of lines
    # A list of degrees of freedom for the Dirichlet boundary conditions
    dirichlet_dofs = jnp.array(D * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1])
    # print('dirichlet_dofs: ', dirichlet_dofs)
    # Global unraveled
    dirichlet_values_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(dirichlet_values)
    # A global vector with 0's where values are not boundary conditions,
    # and 1's corresponding to Dirichlet BCs.
    dirichlet_mask_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(1.0)

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
    # If all of the checks are true, then we can JIT compile the functions
    is_homogeneous = (
        is_x_homogeneous
        and is_dphi_dxi_homogeneous
        and is_W_homogeneous
        and is_fe_type_homogeneous
        and is_conn_homogeneous
        and is_mat_params_homogeneous
    )

    inner_solve = solve_linear_step
    if is_homogeneous:
        print("Batches are homogeneous, using JIT compilation for solve_linear_step")
        inner_solve = jax.jit(solve_linear_step, static_argnames="linear_solver_type")

    # capture memory usage before
    if profile_memory:
        start_memory_profile("solve_linear_step")

    u, new_internal_state_beqi, residual, relative_error, info = inner_solve(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_0_g=u_0_g,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=jnp.array(dirichlet_values),
        linear_solver_type=linear_solver_type,
    )

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
        plt.title(f"Residual History During Iteration Using {linear_solver_type}")
        plt.xlabel("iteration")
        plt.ylabel("|R|")
        plt.yscale("log")
        plt.show()

    for i in range(len(element_batches)):
        element_batches[i].internal_state_eqi = new_internal_state_beqi[i]

    return (u, residual, new_internal_state_beqi,element_batches)

def solve_nonlinear_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.float32]],
    element_batches: list[ElementBatch],
    element_residual_func: Callable,
    u_0_g: jnp.ndarray | None,
    dirichlet_bcs: np.ndarray[Any, np.dtype[np.uint64]],
    dirichlet_values: np.ndarray[Any, np.dtype[np.float32]],
    linear_solver_type: LinearSolverType = LinearSolverType.DIRECT_INVERSE_JNP,
    plot_convergence: bool = False,
    profile_memory: bool = False,
):
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
    profile_memory     : indicates if GPU memory usage should be profiled, which will create *.prof
                         files in the current directory

    Returns
    -------
    u  : solution (displacement), ndarray[float, (V * D)]
    R  : residual vector evaluated at the solution, ndarray[float, (V * D)]
    """

    B = len(element_batches)
    V = vertices_vd.shape[0]
    D = vertices_vd.shape[1]

    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(V * D,))

    assert D <= 3
    assert u_0_g.shape == (V * D,)
    assert dirichlet_bcs.shape[0] <= D * V
    assert dirichlet_bcs.shape[1] == 2
    assert dirichlet_values.shape[0] == dirichlet_bcs.shape[0]
    for b in element_batches:
        assert b.connectivity_en.shape[0] == b.material_params_eqm.shape[0]
        assert b.connectivity_en.shape[1] <= V

    # For each batch find the list of vertices that are unique to form a local
    # numbering for each batch.
    # batch_to_global_map = [np.unique(b.connectivity_en) for b in element_batches]

    # Setup inputs
    # Wrap the provided callables to be compatible with jit
    element_residual_func = jax.tree_util.Partial(element_residual_func)
    constitutive_model_b = [
        jax.tree_util.Partial(b.constitutive_model) for b in element_batches
    ]
    xi_bqp, W_bq = zip(*[get_quadrature(fe_type=b.fe_type) for b in element_batches])
    phi_bqn, dphi_dxi_bqnp = zip(
        *[
            eval_basis_and_derivatives(fe_type=b.fe_type, xi_qp=xi_bqp[i])
            for i, b in enumerate(element_batches)
        ]
    )

    material_params_beqm = [b.material_params_eqm for b in element_batches]
    internal_state_beqi = [b.internal_state_eqi for b in element_batches]
    x_bend = [
        mesh_to_jax(vertices=vertices_vd, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # TODO JIT this group of lines
    # A list of degrees of freedom for the Dirichlet boundary conditions
    dirichlet_dofs = jnp.array(D * dirichlet_bcs[:, 0] + dirichlet_bcs[:, 1])
    # print('dirichlet_dofs: ', dirichlet_dofs)
    # Global unraveled
    dirichlet_values_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(dirichlet_values)
    # A global vector with 0's where values are not boundary conditions,
    # and 1's corresponding to Dirichlet BCs.
    dirichlet_mask_g = jnp.zeros_like(u_0_g).at[dirichlet_dofs].set(1.0)

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
    # If all of the checks are true, then we can JIT compile the functions
    is_homogeneous = (
        is_x_homogeneous
        and is_dphi_dxi_homogeneous
        and is_W_homogeneous
        and is_fe_type_homogeneous
        and is_conn_homogeneous
        and is_mat_params_homogeneous
    )

    inner_solve = solve_nonlinear_step
    if is_homogeneous:
        print("Batches are homogeneous, using JIT compilation for solve_linear_step")
        inner_solve = jax.jit(solve_nonlinear_step, static_argnames="nonlinear_solver_type")

    # capture memory usage before
    if profile_memory:
        start_memory_profile("solve_linear_step")

    u, new_internal_state_beqi, residual, relative_error, info = inner_solve(
        element_residual_func=element_residual_func,
        constitutive_model_b=constitutive_model_b,
        material_params_beqm=material_params_beqm,
        internal_state_beqi=internal_state_beqi,
        x_bend=x_bend,
        dphi_dxi_bqnp=dphi_dxi_bqnp,
        W_bq=W_bq,
        assembly_map_b=assembly_map_b,
        u_0_g=u_0_g,
        dirichlet_values_g=dirichlet_values_g,
        dirichlet_mask_g=dirichlet_mask_g,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=jnp.array(dirichlet_values),
        linear_solver_type=linear_solver_type,
    )

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
        plt.title(f"Residual History During Iteration Using {linear_solver_type}")
        plt.xlabel("iteration")
        plt.ylabel("|R|")
        plt.yscale("log")
        plt.show()  

    for i in range(len(element_batches)):
        element_batches[i].internal_state_eqi = new_internal_state_beqi[i]


    return (u, residual, new_internal_state_beqi,element_batches)
