import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from .utils import rank2_tensor_to_voigt, rank2_voigt_to_tensor, is_required


@jax.jit
def elastic_isotropic(eps_dd: jnp.ndarray, material_params_m: jnp.ndarray):
    """
    A constitive relation for a linear elastic isotropic material.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    material_params_m : material parameters, ndarray[float, (M,)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """

    E = material_params_m[..., 0]
    nu = material_params_m[..., 1]
    G = 0.5 * E / (1.0 + nu)
    if eps_dd.shape[1] == 1:  # 1D
        C_ss = E
    elif eps_dd.shape[1] == 2:  # 2D
        C_ss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, 0.0],
                    [-nu / E, 1.0 / E, 0.0],
                    [0.0, 0.0, 1.0 / G],
                ]
            )
        )
    elif eps_dd.shape[1] == 3:  # 3D
        C_ss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, -nu / E, 0.0, 0.0, 0.0],
                    [-nu / E, 1.0 / E, -nu / E, 0.0, 0.0, 0.0],
                    [-nu / E, -nu / E, 1.0 / E, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0 / G, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0 / G, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G],
                ]
            )
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_dd = rank2_voigt_to_tensor(
        jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    )
    return stress_dd, jnp.array([]) # no internal state


@jax.jit
def elastic_orthotropic(
    eps_dd: jnp.ndarray, material_params_m: jnp.ndarray
):
    """
    A constitive relation for a linear elastic orthotropic material.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    mat_params_m : material parameters, ndarray[float, (M,)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """
    if eps_dd.shape[1] == 1:  # 1D
        assert (
            material_params_m.shape[-1] == 1
        ), f"Orthotropic elasticity in 1D requires 1 material parameter, received {material_params_m.shape[-1]}"

        C_ss = material_params_m[0]

    elif eps_dd.shape[1] == 2:  # 2D
        assert (
            material_params_m.shape[-1] == 4
        ), f"Orthotropic elasticity in 2D requires 4 material parameters, received {material_params_m.shape[-1]}"

        E_xx = material_params_m[0]
        E_yy = material_params_m[1]
        nu_xy = material_params_m[2]
        G_xy = material_params_m[3]

        C_ss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E_xx, -nu_xy / E_xx, 0.0],
                    [-nu_xy / E_xx, 1.0 / E_yy, 0.0],
                    [0.0, 0.0, 1.0 / G_xy],
                ]
            )
        )

    elif eps_dd.shape[1] == 3:  # 3D
        assert (
            material_params_m.shape[-1] == 9
        ), f"Orthotropic elasticity in 3D requires 9 material parameters, received {material_params_m.shape[-1]}"

        E_xx = material_params_m[0]
        E_yy = material_params_m[1]
        E_zz = material_params_m[2]
        nu_xy = material_params_m[3]
        nu_yz = material_params_m[4]
        nu_xz = material_params_m[5]
        G_xy = material_params_m[6]
        G_yz = material_params_m[7]
        G_xz = material_params_m[8]

        # Note: inv could be avoided if it is a bottleneck, see:
        # https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
        C_ss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E_xx, -nu_xy / E_xx, -nu_xz / E_xx, 0.0, 0.0, 0.0],
                    [-nu_xy / E_xx, 1.0 / E_yy, -nu_yz / E_yy, 0.0, 0.0, 0.0],
                    [-nu_xz / E_xx, -nu_yz / E_yy, 1.0 / E_zz, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0 / G_yz, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0 / G_xz, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G_xy],
                ]
            )
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_dd = rank2_voigt_to_tensor(
        jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    )
    return stress_dd, jnp.array([]) # no internal state


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
    material_params : material parameters, ndarray[float, (Q, M)] or ndarray[float, (M,)]
    constitutive_relation : constitutive stress-strain relation, function with arguments
                  (eps_dd: jnp.ndarray, material_params: jnp.ndarray)

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

    constitutive_args = []
    in_axes = []

    if is_required(constitutive_model, "eps_dd"):
        constitutive_args.append(eps_qdd)
        in_axes.append(0)

    if is_required(constitutive_model, "material_params_m"):
        constitutive_args.append(material_params)
        if material_params.ndim == 1:
            in_axes.append(None)
        else:
            in_axes.append(0)

    if is_required(constitutive_model, "internal_state_i"):
        constitutive_args.append(internal_state_i)
        in_axes.append(0)

    constitutive_model_vmap = jax.vmap(constitutive_model, in_axes=tuple(in_axes))
    stress_qdd, new_internal_state_qi = constitutive_model_vmap(*constitutive_args)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd, new_internal_state_qi
