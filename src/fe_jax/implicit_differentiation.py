import jax
import jax.numpy as jnp
from functools import partial
import jax.experimental.sparse as jsparse

from .sparse_linear_solve import LinearSolverType
from .fea import (
    solve_nonlinear_quasi_step,
    calculate_jacobian_wo_constraints,
    calculate_residual_w_constraints,
    compute_ISV_be
)
from .sparse_matrix import apply_dirichlet_bcs_lhs

@partial(jax.custom_vjp, nondiff_argnums=(3,))
def solve_nonlinear_quasi_step_autodiff_vjp(
    element_residual_func, ebc, assembly_map_b, jacobian_nnz, u_0_g, constraints, solver_options, f_ext
):
    """
    Solve one nonlinear quasi-static step with a custom implicit VJP.

    Parameters
    ----------
    element_residual_func : element residual function
    ebc                   : element boundary-condition and internal-state data
    assembly_map_b        : element-to-global assembly map
    jacobian_nnz          : number of nonzero Jacobian entries
    u_0_g                 : initial guess for the global solution
    constraints           : global degree-of-freedom constraints
    solver_options        : nonlinear and linear solver options
    f_ext                 : external force vector

    Returns
    -------
    u_f            : converged global solution
    ISV_be         : element-batch internal state variables
    R_f            : final residual
    relative_error : nonlinear solver convergence error
    info           : solver information

    Notes
    -----
    The custom VJP uses implicit differentiation at the converged
    equilibrium state, avoiding differentiation through the nonlinear
    solver iterations.
    """
    return solve_nonlinear_quasi_step(
        element_residual_func, ebc, assembly_map_b, jacobian_nnz, u_0_g, constraints, solver_options, f_ext
    )

def _solve_fwd(element_residual_func, ebc, assembly_map_b, jacobian_nnz, u_0_g, constraints, solver_options, f_ext):
    outputs = solve_nonlinear_quasi_step(
        element_residual_func, ebc, assembly_map_b, jacobian_nnz, u_0_g, constraints, solver_options, f_ext
    )
    u_f, ISV_be, R_f, relative_error, info = outputs
    
    # Pre-compute explicitly assembled Jacobian for the backward pass
    J_f = calculate_jacobian_wo_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        precomputed_jacobian_nnz=jacobian_nnz,
        u_f=u_f,
    )
    J_f_constrained = apply_dirichlet_bcs_lhs(J_f, constraints.dep_dofs)
    
    res = (element_residual_func, ebc, assembly_map_b, jacobian_nnz, u_f, constraints, solver_options, f_ext, J_f_constrained)
    return outputs, res

def _solve_bwd(jacobian_nnz, res, g_outputs):
    element_residual_func, ebc, assembly_map_b, jacobian_nnz, u_f, constraints, solver_options, f_ext, J_f_constrained = res
    g_u_f, g_new_internal_state_beqi, g_R_f, g_rel_error, g_info = g_outputs
    
    def forward_for_vjp(u_local, ebc_local, constraints_local):
        R_f_local, new_internal_state_beqi = calculate_residual_w_constraints(
            u_f=u_local,
            element_residual_func=element_residual_func,
            ebc=ebc_local,
            assembly_map_b=assembly_map_b,
            constraints=constraints_local,
            f_ext=f_ext,
        )
        return R_f_local, new_internal_state_beqi

    primals_out, vjp_fn = jax.vjp(forward_for_vjp, u_f, ebc, constraints)
    R_f_out, new_internal_state_beqi_out = primals_out
    
    zero_R_f = jnp.zeros_like(R_f_out)
    
    def sanitize_cotangent(g, p):
        if type(g) is jax.custom_derivatives.SymbolicZero or g is None:
            if hasattr(p, "shape"):
                return jnp.zeros_like(p)
            return p
        return g

    if g_new_internal_state_beqi is None or type(g_new_internal_state_beqi) is jax.custom_derivatives.SymbolicZero:
        clean_g_new_internal_state_beqi = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p) if hasattr(p, "shape") else p, new_internal_state_beqi_out)
    else:
        clean_g_new_internal_state_beqi = jax.tree_util.tree_map(sanitize_cotangent, g_new_internal_state_beqi, new_internal_state_beqi_out)
    
    def A(lambda_R):
        return J_f_constrained.T @ lambda_R
        
    rhs_vjp = vjp_fn((zero_R_f, clean_g_new_internal_state_beqi))[0]
    
    if type(g_u_f) is jax.custom_derivatives.SymbolicZero or g_u_f is None:
        g_u_f_val = jnp.zeros_like(u_f)
    else:
        g_u_f_val = g_u_f
        
    rhs = -g_u_f_val - rhs_vjp
    
    if solver_options.linear_solve_type == LinearSolverType.DENSE_INVERSE_JNP:
        def R_w_dirichlet(u_local):
            R, _ = calculate_residual_w_constraints(
                u_f=u_local,
                element_residual_func=element_residual_func,
                ebc=ebc,
                assembly_map_b=assembly_map_b,
                constraints=constraints,
                f_ext=f_ext,
            )
            return R
        J_dense = jax.jacfwd(R_w_dirichlet)(u_f)
        lambda_R = jnp.linalg.solve(J_dense.T, rhs)
    else:
        lambda_R, _ = jax.scipy.sparse.linalg.gmres(A, rhs, tol=1e-6, maxiter=100)
    
    
    _, g_ebc, g_constraints = vjp_fn((lambda_R, clean_g_new_internal_state_beqi))
    
    g_u_0_g = jnp.zeros_like(u_f)
    
    return None, g_ebc, None, g_u_0_g, g_constraints, None, None

solve_nonlinear_quasi_step_autodiff_vjp.defvjp(_solve_fwd, _solve_bwd)
