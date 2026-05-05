import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from .utils import (
    rank2_tensor_to_voigt,
    rank2_voigt_to_tensor,
    is_required,
    debug_print,
)


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
        C_ss = jnp.array([[E]])
    elif eps_dd.shape[1] == 2:  # 2D
        S_ss = jnp.array(
            [
                [1.0 / E, -nu / E, 0.0],
                [-nu / E, 1.0 / E, 0.0],
                [0.0, 0.0, 1.0 / G],
            ]
        )
        #C_ss = jnp.linalg.inv(S_ss)
        C_ss = E/((1.0-2.0*nu)*(1.0+nu))*jnp.array(
            [
                [1.0-nu, nu    , 0.0            ],
                [nu    , 1.0-nu, 0.0            ],
                [0.0   , 0.0   , (1.0-2.0*nu)/nu]
            ]
        )
    elif eps_dd.shape[1] == 3:  # 3D
        S_ss = jnp.array(
            [
                [1.0 / E, -nu / E, -nu / E, 0.0, 0.0, 0.0],
                [-nu / E, 1.0 / E, -nu / E, 0.0, 0.0, 0.0],
                [-nu / E, -nu / E, 1.0 / E, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0 / G, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0 / G, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G],
            ]
        )
        #C_ss = jnp.linalg.inv(S_ss)
        C_ss = E/((1.0-2.0*nu)*(1.0+nu))*jnp.array(
            [
                [1.0-nu, nu    , nu    , 0.0            , 0.0             , 0.0             ],
                [nu    , 1.0-nu, nu    , 0.0            , 0.0             , 0.0             ],
                [nu    , nu    , 1.0-nu, 0.0            , 0.0             , 0.0             ],
                [0.0   , 0.0   , 0.0   , (1.0-2.0*nu)/nu, 0.0             , 0.0             ],
                [0.0   , 0.0   , 0.0   , 0.0            , (1.0-2.0*nu)/2.0, 0.0             ],
                [0.0   , 0.0   , 0.0   , 0.0            , 0.0             , (1.0-2.0*nu)/2.0]
            ]
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_dd = rank2_voigt_to_tensor(
        jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    )
    return stress_dd, jnp.array([])  # no internal state


@jax.jit
def elastic_orthotropic(eps_dd: jnp.ndarray, material_params_m: jnp.ndarray):
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

        S_ss = jnp.array(
            [
                [1.0 / E_xx, -nu_xy / E_xx, 0.0],
                [-nu_xy / E_xx, 1.0 / E_yy, 0.0],
                [0.0, 0.0, 1.0 / G_xy],
            ]
        )
        #C_ss = jnp.linalg.inv(S_ss)
        # For the direct def'n of C_ss we also need nu_yx, nu_zy, and nu_zx.
        # These can be determined from the above parameters along with symmetry. 
        nu_yx = E_yy/E_xx*nu_xy

        Delta =  (1 - nu_xy*nu_yx)/(E_xx*E_yy)
        C_ss = (1/Delta)*jnp.array(
            [
                [1.0/E_yy  , nu_yx/E_yy, 0.0       ],
                [nu_xy/E_xx, 1.0/E_xx  , 0.0       ],
                [0.0       , 0.0       , Delta*G_xy]
            ]
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
        G_xz = material_params_m[9]
        # Note: inv could be avoided if it is a bottleneck, see:
        # https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
        S_ss = jnp.array(
            [
                [1.0 / E_xx, -nu_xy / E_xx, -nu_xz / E_xx, 0.0, 0.0, 0.0],
                [-nu_xy / E_xx, 1.0 / E_yy, -nu_yz / E_yy, 0.0, 0.0, 0.0],
                [-nu_xz / E_xx, -nu_yz / E_yy, 1.0 / E_zz, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0 / G_yz, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0 / G_xz, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G_xy],
            ]
        )
        # For the direct def'n of C_ss we also need nu_yx, nu_zy, and nu_zx.
        # These can be determined from the above parameters along with symmetry. 
        nu_yx = E_yy/E_xx*nu_xy
        nu_zy = E_zz/E_yy*nu_yz
        nu_zx = E_zz/E_xx*nu_xz

        Delta =  (1 - nu_xy*nu_yx - nu_yz*nu_zy - nu_xz*nu_zx - 2*nu_xy*nu_yz*nu_zx)/(E_xx*E_yy*E_zz)
        #C_ss = jnp.linalg.inv(S_ss)
        C_ss = (1/Delta)*jnp.array(
            [
                [(1.0-nu_yz*nu_zy)/(E_yy*E_zz)  , (nu_yx+nu_zx*nu_yz)/(E_yy*E_zz), (nu_zx+nu_yx*nu_zy)/(E_yy*E_zz), 0.0, 0.0, 0.0],
                [(nu_xy+nu_xz*nu_zy)/(E_xx*E_zz), (1.0-nu_zx*nu_xz)/(E_xx*E_zz)  , (nu_zy+nu_zx*nu_xy)/(E_xx*E_zz), 0.0, 0.0, 0.0],
                [(nu_xz+nu_xy*nu_yz)/(E_xx*E_yy), (nu_yz+nu_xz*nu_yx)/(E_xx*E_yy), (1.0-nu_xy*nu_yx)/(E_xx*E_yy)  , 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0 , Delta*G_yz ,0.0       , 0.0],
                [0.0, 0.0, 0.0 , 0.0        ,Delta*G_xz, 0.0],
                [0.0, 0.0, 0.0 , 0.0        ,0.0       , Delta*G_xy],
            ]
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_dd = rank2_voigt_to_tensor(
        jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    )
    return stress_dd, jnp.array([])  # no internal state


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
    assert (
        P == D
    ), f"Number of dimensions in the parametric coordinate system of the element must match the dimension of the problem, {P} != {D}"
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
        constitutive_args.append(internal_state_qi)
        in_axes.append(0)

    constitutive_model_vmap = jax.vmap(constitutive_model, in_axes=tuple(in_axes))
    stress_qdd, new_internal_state_qi = constitutive_model_vmap(*constitutive_args)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd, new_internal_state_qi

@jax.jit
def elastic_truss(eps_dd: jnp.ndarray, material_params_m: jnp.ndarray, x_nd: jnp.ndarray, u_nd=jnp.ndarray):
    """
    A constitive relation for a an elastic truss.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    material_params_m : material parameters, ndarray[float, (M,)]
    x_nd          : coordinates, ndarray[float, (N, D)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """

    E = material_params_m[..., 0]
    A = material_params_m[..., 1]
    # Assumes the node number puts the endpoints as first and last entries. 
    dx_d = (x_nd+u_nd)[-1,:]-(x_nd+u_nd)[0,:]
    l_d = dx_d/jnp.sqrt(jnp.dot(dx_d,dx_d))

    P_dd = jnp.outer(l_d,l_d)
    eps_a = jnp.einsum("i,ij,j->", l_d, eps_dd, l_d)
    stress_dd = E*A*eps_a*P_dd
    # if eps_dd.shape[1] == 1:  # 1D
    #     C_ss = jnp.array([[E*A/L]])
    # elif eps_dd.shape[1] == 2:  # 2D
    #     C_ss =  jnp.array(
    #         # THIS IS WRONG!, should be 3x3, not sure where I went wrong.
    #         [
    #             [ dx_d[0]*dx_d[0], dx_d[0]*dx_d[1],-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1]],
    #             [ dx_d[0]*dx_d[1], dx_d[1]*dx_d[1],-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1]],
    #             [-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1], dx_d[0]*dx_d[0], dx_d[0]*dx_d[1]],
    #             [-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1], dx_d[0]*dx_d[1], dx_d[1]*dx_d[1]],
    #         ]
    #     )
    # elif eps_dd.shape[1] == 3:  # 3D
    #     C_ss = (E*A/L**3)*jnp.array(
    #         [
    #             [ dx_d[0]*dx_d[0], dx_d[0]*dx_d[1], dx_d[0]*dx_d[2],-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1],-dx_d[0]*dx_d[2]],
    #             [ dx_d[0]*dx_d[1], dx_d[1]*dx_d[1], dx_d[1]*dx_d[2],-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1],-dx_d[1]*dx_d[2]],
    #             [ dx_d[0]*dx_d[2], dx_d[1]*dx_d[2], dx_d[2]*dx_d[2],-dx_d[0]*dx_d[2],-dx_d[1]*dx_d[2],-dx_d[2]*dx_d[2]],
    #             [-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1],-dx_d[0]*dx_d[2], dx_d[0]*dx_d[0], dx_d[0]*dx_d[1], dx_d[0]*dx_d[2]],
    #             [-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1],-dx_d[1]*dx_d[2], dx_d[0]*dx_d[1], dx_d[1]*dx_d[1], dx_d[1]*dx_d[2]],
    #             [-dx_d[0]*dx_d[2],-dx_d[1]*dx_d[2],-dx_d[2]*dx_d[2], dx_d[0]*dx_d[2], dx_d[1]*dx_d[2], dx_d[2]*dx_d[2]],
    #         ]
    #     )
    # else:
    #     raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    # stress_dd = rank2_voigt_to_tensor(
    #     jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    # )
    return stress_dd, jnp.array([])  # no internal state



@jax.jit
def linear_truss_residual(
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
    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    det_J_q = jnp.sqrt(jnp.linalg.det(jnp.einsum("qpd,qrd->qpr",J_qpd,J_qpd)))
    def lstsq_one(J_pd,dphi_dxi_np):
        dphi_dx_nd = jnp.linalg.lstsq(J_pd, dphi_dxi_np.T)[0]
        return dphi_dx_nd.T
    
    dphi_dx_qnd = jax.vmap(lstsq_one, in_axes=(0,0))(J_qpd,dphi_dxi_qnp)

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
        constitutive_args.append(internal_state_qi)
        in_axes.append(0)

    if is_required(constitutive_model, "x_nd"):
        constitutive_args.append(x_nd)
        in_axes.append(None)

    if is_required(constitutive_model, "u_nd"):
        constitutive_args.append(u_nd)
        in_axes.append(None)

    constitutive_model_vmap = jax.vmap(constitutive_model, in_axes=tuple(in_axes))
    stress_qdd, new_internal_state_qi = constitutive_model_vmap(*constitutive_args)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    # jax.debug.print('x_nd = \n{}\nu_nd = \n{}\nR_nd = \n{}\n',x_nd,u_nd,R_nd)
    # jax.debug.print('x_nd = \n{}\n',x_nd)
    return R_nd, new_internal_state_qi

def stiff_matrix(material_params_m: jnp.ndarray, x_nd: jnp.ndarray):
    """
    Compute the stiffness matrix directly from John Whitcomb's notes.

    Parameters
    ----------
    material_params_m : material parameters, ndarray[float, (M,)]
    x_nd          : coordinates, ndarray[float, (N, D)]

    Returns
    -------
    K_global  : stiffness matrix, ndarray[float, (D, D)]
    """

    E = material_params_m[..., 0]
    A = material_params_m[..., 1]
    # Assumes the node number puts the endpoints as first and last entries. 
    dx_d = x_nd[-1,:]-x_nd[0,:]
    L = jnp.linalg.norm(dx_d)

    if x_nd.shape[1] == 1:  # 1D
        K_global = E*A/L*jnp.array([[1,-1],[-1,1]])
    elif x_nd.shape[1] == 2:  # 2D
        K_global =  (E*A/L**3)*jnp.array(
            [
                [ dx_d[0]*dx_d[0], dx_d[0]*dx_d[1],-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1]],
                [ dx_d[0]*dx_d[1], dx_d[1]*dx_d[1],-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1]],
                [-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1], dx_d[0]*dx_d[0], dx_d[0]*dx_d[1]],
                [-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1], dx_d[0]*dx_d[1], dx_d[1]*dx_d[1]],
            ]
        )
    elif x_nd.shape[1] == 3:  # 3D
        K_global = (E*A/L**3)*jnp.array(
            [
                [ dx_d[0]*dx_d[0], dx_d[0]*dx_d[1], dx_d[0]*dx_d[2],-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1],-dx_d[0]*dx_d[2]],
                [ dx_d[0]*dx_d[1], dx_d[1]*dx_d[1], dx_d[1]*dx_d[2],-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1],-dx_d[1]*dx_d[2]],
                [ dx_d[0]*dx_d[2], dx_d[1]*dx_d[2], dx_d[2]*dx_d[2],-dx_d[0]*dx_d[2],-dx_d[1]*dx_d[2],-dx_d[2]*dx_d[2]],
                [-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1],-dx_d[0]*dx_d[2], dx_d[0]*dx_d[0], dx_d[0]*dx_d[1], dx_d[0]*dx_d[2]],
                [-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1],-dx_d[1]*dx_d[2], dx_d[0]*dx_d[1], dx_d[1]*dx_d[1], dx_d[1]*dx_d[2]],
                [-dx_d[0]*dx_d[2],-dx_d[1]*dx_d[2],-dx_d[2]*dx_d[2], dx_d[0]*dx_d[2], dx_d[1]*dx_d[2], dx_d[2]*dx_d[2]],
            ]
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    return K_global


@jax.jit
def stiffness_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: Callable,
    # u_nd: jnp.ndarray,
    # x_nd: jnp.ndarray,
    # material_params: jnp.ndarray,
    # internal_state_qi: jnp.ndarray,
):
    """
    Residual function that computes the residual for the weak form corresponding to linear
    elasticity using the stiffness matrix.

    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    material_params : material parameters, ndarray[float, (Q, M)] or ndarray[float, (M,)]
                  (eps_dd: jnp.ndarray, material_params: jnp.ndarray)

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """
    E = material_params[..., 0]
    A = material_params[..., 1]
    dx = x_nd[1,:]-x_nd[0,:]
    L = jnp.linalg.norm(dx) 

    T= jnp.vstack((jnp.hstack((dx/L,0*dx)),jnp.hstack((0*dx,dx/L))))
    K = E*A/L*jnp.array([[1,-1],[-1,1]])
    K_global = jnp.einsum("ni,ij,jm->nm",T.T,K,T)
    #K_direct = stiff_matrix(material_params_m=material_params,x_nd=x_nd)

    #assert jnp.isclose(K_direct,K_global).all(), "Computing stiffness matrix from T^T K T and direct element-by-element implementation do not match."
    # jax.debug.print('x_nd = \n{}\nu_nd = \n{}\nK_global = \n{}\n',x_nd,u_nd,K_global)
    R_nd = jnp.einsum("dk,k->d",K_global,u_nd.reshape((-1))).reshape((-1,x_nd.shape[1]))

    # jax.debug.print('x_nd = \n{}\nu_nd = \n{}\nR_nd = \n{}\n',x_nd,u_nd,R_nd)
    # jax.debug.print('x_nd = \n{}\n',x_nd)
    new_internal_state_qi = internal_state_qi
    return R_nd, new_internal_state_qi

