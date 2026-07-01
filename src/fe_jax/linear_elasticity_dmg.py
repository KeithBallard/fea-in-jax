import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from .utils import rank2_tensor_to_voigt, rank2_voigt_to_tensor, is_required
from .dmg_eqs_vmap import *


@jax.jit
def update_one_q(eps_dd, state_i, mat_param_m, C_ss):
        # Extract state
        prev_eps    = state_i[:3]
        prev_stress = state_i[3:6]
        d, Y, tau0, vM0, dt = state_i[6:]
        # === DAMAGE ===
        dmg_crit = calc_dmg_crit(prev_eps, mat_param_m)
        update_Y, update_d, update_tau0 = jax.lax.cond(
            dmg_crit > 0,
            lambda op: calc_d(*op),
            lambda op: (op[0], op[1], op[2]),
            operand=(Y, d, tau0, dt, mat_param_m, prev_eps,C_ss)
        )

        # === YIELD ===
        yield_crit = calc_yield_crit(prev_stress, mat_param_m)
        update_vM0, delta_C = jax.lax.cond(
            yield_crit > 0,
            lambda op: calc_update_Dmat(*op),
            lambda op: (op[1], jnp.zeros_like(C_ss)),
            operand=(mat_param_m, vM0, prev_stress, C_ss)
        )

        # === STRESS ===
        eps_vogit = rank2_tensor_to_voigt(eps_dd)
        d_eps_vogit = eps_vogit - prev_eps

        d_stress_vogit = (C_ss + delta_C) @ d_eps_vogit
        stress_vogit = prev_stress + d_stress_vogit
        stress_dmg = (1 - d) * stress_vogit
        stress_dd = rank2_voigt_to_tensor(stress_dmg)

        # === Update state ===
        update_state_i = state_i.copy()
        update_state_i = update_state_i.at[0:3].set(eps_vogit)
        update_state_i = update_state_i.at[3:6].set(stress_vogit)
        update_state_i = update_state_i.at[6].set(update_d)
        update_state_i = update_state_i.at[7].set(update_Y)
        update_state_i = update_state_i.at[8].set(update_tau0)
        update_state_i = update_state_i.at[9].set(update_vM0)

        return stress_dd, update_state_i

@jax.jit
def damage_elastic_isotropic_vmap(eps_qdd: jnp.ndarray, material_params: jnp.ndarray, internal_state_qi: jnp.ndarray):
    """
    A constitive relation for a linear elastic isotropic material.

    Parameters
    ----------
    eps_qdd       : infinitesimal strain tensor, ndarray[float, (Q, D, D)]
    mat_params : material parameters, ndarray[float, (Q, M)]

    Returns
    -------
    stress_qdd  : stress tensor, ndarray[float, (Q, D, D)]
    """

    E  = material_params[..., 0]
    nu = material_params[..., 1]
    zero = jnp.zeros_like(nu)
    if eps_qdd.shape[2] == 1:  # 1D
        C_qss = E.transpose((1, 0))[:, jnp.newaxis]
    elif eps_qdd.shape[2] == 2:  # 2D
        # Plane strain
        coef = E / ((1 + nu) * (1 - 2 * nu))
        C_qss = (coef * jnp.array(
                [
                    [1 - nu, nu,     zero],
                    [nu,     1 - nu, zero],
                    [zero,   zero,   (1 - 2 * nu) / 2.0]
                ]
            )).transpose((2, 0, 1))  # if batched

        # Plane stress (Chennie thinks it is, I was wrong!!!! This was tested. Left here for future references. Plane strain is correct)
        # G = 0.5 * E / (1.0 + nu)
        # C_qss = jnp.linalg.inv(
        #     jnp.array(
        #         [
        #             [1.0 / E, -nu / E, zero],
        #             [-nu / E, 1.0 / E, zero],
        #             [zero, zero, 1.0 / G],
        #         ]
        #     ).transpose((2, 0, 1))
        # )

    elif eps_qdd.shape[2] == 3:  # 3D
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                    [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                    [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                    [zero, zero, zero, 1.0 / G, zero, zero],
                    [zero, zero, zero, zero, 1.0 / G, zero],
                    [zero, zero, zero, zero, zero, 1.0 / G],
                ]
            ).transpose((2, 0, 1))
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    update_stress_qdd, update_internal_state_qi = jax.vmap(update_one_q)(
                                                            eps_qdd,                # (Q, 2, 2)
                                                            internal_state_qi,      # (Q, i)
                                                            material_params,     # (Q, M)
                                                            C_qss)                  # (Q, 3,3)
    return update_stress_qdd, update_internal_state_qi


@jax.jit
def elastic_orthotropic(
    eps_qdd: jnp.ndarray, material_params: jnp.ndarray, internal_state_qi: jnp.ndarray
):
    """
    A constitive relation for a linear elastic isotropic material.

    Parameters
    ----------
    eps_qdd       : infinitesimal strain tensor, ndarray[float, (Q, D, D)]
    material_params : material parameters, ndarray[float, (Q, M)]

    Returns
    -------
    stress_qdd  : stress tensor, ndarray[float, (Q, D, D)]
    """
    zero = jnp.zeros(shape=(material_params.shape[0:1]))
    if eps_qdd.shape[2] == 1:  # 1D
        assert (
            material_params.shape[-1] == 1
        ), f"Orthotropic elasticity in 1D requires 1 material parameter, received {material_params.shape[-1]}"

        E = material_params[..., 0]
        C_qss = E.transpose((1, 0))[:, jnp.newaxis]

    elif eps_qdd.shape[2] == 2:  # 2D
        # assert (
        #     material_params.shape[-1] == 4
        # ), f"Orthotropic elasticity in 2D requires 4 material parameters, received {material_params.shape[-1]}"

        # E_xx  = material_params[..., 0]
        # E_yy  = material_params[..., 1]
        # nu_xy = material_params[..., 2]
        # G_xy  = material_params[..., 3]
        # G_23  = material_params[..., 4]

        # C_qss = jnp.linalg.inv(
        #     jnp.array(
        #         [
        #             [1.0 / E_xx, -nu_xy / E_xx, zero],
        #             [-nu_xy / E_xx, 1.0 / E_yy, zero],
        #             [zero, zero, 1.0 / G_xy],
        #         ]
        #     ).transpose((2, 0, 1))
        # )


        # Unpack material parameters from batch 0
        E1, E2, nu12, G12, G23 = material_params[0]

        # Transverse isotropy parameters
        E3 = E2
        nu13 = nu12
        nu23 = E2 / (2.0 * G23) - 1.0

        delta = (
            1.0
            - 2.0 * (E3 / E1) * nu12 * nu13 * nu23
            - (E3 / E1) * nu13**2
            - (E3 / E2) * nu23**2
            - (E2 / E1) * nu12**2
        )

        # Plane strain Dmat (Voigt: xx, yy, xy)
        C11 = E1 * (1.0 - (E3 / E2) * nu23**2) / delta
        C22 = E2 * (1.0 - (E3 / E1) * nu13**2) / delta
        C12 = (E2 * nu12 + E3 * nu13 * nu23) / delta
        C66 = G12
        C_ss = jnp.array([
            [C11, C12, 0.0],
            [C12, C22, 0.0],
            [0.0,  0.0, C66],
        ])

        C_qss = jnp.ones(shape=(material_params.shape[0],3,3)) * C_ss[None,:,:]

    elif eps_qdd.shape[2] == 3:  # 3D
        assert (
            material_params.shape[-1] == 9
        ), f"Orthotropic elasticity in 3D requires 9 material parameters, received {material_params.shape[-1]}"

        E_xx  = material_params[..., 0]
        E_yy  = material_params[..., 1]
        E_zz  = material_params[..., 2]
        nu_xy = material_params[..., 3]
        nu_yz = material_params[..., 4]
        nu_xz = material_params[..., 5]
        G_xy  = material_params[..., 6]
        G_yz  = material_params[..., 7]
        G_xz  = material_params[..., 8]

        # Note: inv could be avoided if it is a bottleneck, see:
        # https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E_xx, -nu_xy / E_xx, -nu_xz / E_xx, zero, zero, zero],
                    [-nu_xy / E_xx, 1.0 / E_yy, -nu_yz / E_yy, zero, zero, zero],
                    [-nu_xz / E_xx, -nu_yz / E_yy, 1.0 / E_zz, zero, zero, zero],
                    [zero, zero, zero, 1.0 / G_yz, zero, zero],
                    [zero, zero, zero, zero, 1.0 / G_xz, zero],
                    [zero, zero, zero, zero, zero, 1.0 / G_xy],
                ]
            ).transpose((2, 0, 1))
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    prev_eps    = internal_state_qi[:,:3]
    eps_vogit   = rank2_tensor_to_voigt(eps_qdd)
    d_eps_vogit = eps_vogit - prev_eps
    
    d_stress_vogit  = jnp.einsum("qsi,qi->qs", C_qss, d_eps_vogit)
    prev_stress  = internal_state_qi[:,3:6]
    stress_vogit    = prev_stress + d_stress_vogit
    stress_qdd   = rank2_voigt_to_tensor(stress_vogit)

    update_internal_state_qi = internal_state_qi.copy()
    update_internal_state_qi = update_internal_state_qi.at[:,0:3].set(eps_vogit)
    update_internal_state_qi = update_internal_state_qi.at[:,3:6].set(stress_vogit)
    return stress_qdd, update_internal_state_qi


@jax.jit
def linear_elasticity_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: Callable,
):
    """
    Residual function that computes the residual for the weak form corresponding to linear
    elasticity.

    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    dphi_dxi_qnp  : derivative of basis functions in parametric coordinate system at
                    quadrature points, ndarray[float, (Q, N, P)]
    W_q           : quadrature weights, ndarray[float, (Q,)]
    mat_params_qm : material parameters, ndarray[float, (Q, M)]
    constitutive_relation : constitutive stress-strain relation, function with arguments
                  (eps_qdd: jnp.ndarray, material_params: jnp.ndarray)

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """
    D = u_nd.shape[1]
    P = dphi_dxi_qnp.shape[2]
    assert P == D
    # Formulation assumes solid elements otherwise a different approach is needed (i.e. shells)

    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

    du_dx_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd)
    eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)))

    constitutive_args = {}

    if is_required(constitutive_model, "u_nd"):
        constitutive_args["u_nd"] = u_nd
        
    if is_required(constitutive_model, "x_nd"):
        constitutive_args["x_nd"] = x_nd

    if is_required(constitutive_model, "eps_qdd"):
        constitutive_args["eps_qdd"] = eps_qdd

    if is_required(constitutive_model, "material_params"):
        constitutive_args["material_params"] = material_params

    if is_required(constitutive_model, "internal_state_qi"):
        constitutive_args["internal_state_qi"] = internal_state_qi

    stress_qdd, new_internal_state_qi = constitutive_model(**constitutive_args)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd, new_internal_state_qi