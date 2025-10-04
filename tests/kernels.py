from helper import *

import jax.numpy as jnp
import jax

import numpy as np

timer_n_calls = 10

@dataclass
class Dimensions:
    N_x: int = 2  # Dimensionality of global coordinate system
    N_xi: int = 2  # Dimensionality of parametric coordinate system
    N_u: int = 2  # Dimensionality of solution
    N_eps: int = 3  # Number of strain components (Voigt notation)
    N_mp: int = 2  # Number of material parameters
    N_qp: int = 4  # Number of quadrature points per element
    N_n: int = 3  # Number of nodes per element
    N_dof: int = 6  # Number of DoFs per element (typically V * U)
    N_gn: int = 0  # Number of nodes in the mesh
    N_ge: int = 0  # Number of elements in the mesh


###################################################################################################
# Kernel 1


@timer(time_jit=True, n_calls=timer_n_calls)
def k1_interp_node_to_quad_tensordot(
    x_n: jnp.ndarray, phi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 1 - Interpolate coordinates from nodes to quadrature points

    Parameters
    ----------
    x_n    : dense 3d-array with shape (N_ge, N_n, N_x)
    phi_qp : dense 2d-array with shape (N_qp, N_n)

    Returns
    -------
    x_qp   : dense 3d-array with shape (N_ge, N_qp, N_x)
    """
    return jnp.tensordot(x_n.transpose((0, 2, 1)), phi_qp.T, axes=1).transpose(
        (0, 2, 1)
    )


@timer(time_jit=True, n_calls=timer_n_calls)
def k1_interp_node_to_quad_tensordot_alt(
    x_n: jnp.ndarray, phi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Alternative Kernel 1: Interpolate coordinates from nodes to quadrature points

    Parameters
    ----------
    x_n    : dense 3d-array with shape (N_ge, N_x, N_n)
    phi_qp : dense 2d-array with shape (N_n, N_qp)

    Returns
    -------
    x_qp   : dense 3d-array with shape (N_ge, N_x, N_qp)
    """
    return jnp.tensordot(x_n, phi_qp, axes=1)


@timer(time_jit=True, n_calls=timer_n_calls)
def k1_interp_node_to_quad_einsum_alt(
    x_n: jnp.ndarray, phi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Alternative Kernel 1: Interpolate coordinates from nodes to quadrature points

    Parameters
    ----------
    x_n    : dense 3d-array with shape (N_ge, N_x, N_n)
    phi_qp : dense 2d-array with shape (N_n, N_qp)

    Returns
    -------
    x_qp   : dense 3d-array with shape (N_ge, N_x, N_qp)
    """
    return jnp.einsum("ijk,km->ijm", x_n, phi_qp)


@timer()
def k1_interp_node_to_quad_sum(
    x_n: jnp.ndarray, phi_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert x_n.shape[0] == dims.N_ge
    assert x_n.shape[1] == dims.N_n
    assert x_n.shape[2] == dims.N_x
    assert phi_qp.shape[0] == dims.N_qp
    assert phi_qp.shape[1] == dims.N_n

    x_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_x))
    for i in range(dims.N_ge):
        x_qp[i] = np.dot(phi_qp, x_n[i])
    return jnp.array(x_qp)


###################################################################################################
# Kernel 3


@timer(time_jit=True, n_calls=timer_n_calls)
def k3_param_to_global_jacobian_tensordot(
    x_n: jnp.ndarray, dphi_dxi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 3 - Calculate Jacobian matrices for parametric to global transformation at quad points

    Parameters
    ----------
    x_n         : dense 3d-array with shape (N_ge, N_n, N_x)
    dphi_dxi_qp : dense 3d-array with shape (N_qp, N_n, N_xi)

    Returns
    -------
    J_qp   : dense 4d-array with shape (N_ge, N_qp, N_xi, N_x)
    """
    # Shape of x_n.transpose       dphi_dxi_qp.transpose
    #         (N_ge, N_x, N_n)        (N_n, N_xi, N_qp)
    return jnp.tensordot(
        x_n.transpose((0, 2, 1)), dphi_dxi_qp.transpose((1, 2, 0)), axes=1
    ).transpose((0, 3, 2, 1))


@timer(time_jit=True, n_calls=timer_n_calls)
def k3_param_to_global_jacobian_tensordot_alt(
    x_n: jnp.ndarray, dphi_dxi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Alternative Kernel 3 - Calculate Jacobian matrices for parametric to global transformation at quad points

    Parameters
    ----------
    x_n         : dense 3d-array with shape (N_ge, N_x, N_n)
    dphi_dxi_qp : dense 3d-array with shape (N_n, N_xi, N_qp)

    Returns
    -------
    J_qp   : dense 4d-array with shape (N_ge, N_x, N_xi, N_qp)
    """
    return jnp.tensordot(x_n, dphi_dxi_qp, axes=1)


@timer()
def k3_param_to_global_jacobian_sum(
    x_n: jnp.ndarray, dphi_dxi_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert x_n.shape[0] == dims.N_ge
    assert x_n.shape[1] == dims.N_n
    assert x_n.shape[2] == dims.N_x
    assert dphi_dxi_qp.shape[0] == dims.N_qp
    assert dphi_dxi_qp.shape[1] == dims.N_n
    assert dphi_dxi_qp.shape[2] == dims.N_xi

    J_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_xi, dims.N_x))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            # For 2D:
            # dphi_dxi_qp[j] is array with shape (N_n, N_xi)
            #            [ dphi_0_dxi  ... dphi_N_n_dxi  ] * [ x_0    y_0   ]
            #            [ dphi_0_deta ... dphi_N_n_deta ]   [     ...      ]
            #                                                [ x_N_x  y_N_x ]
            # Result: [  dx_dxi  dy_dxi ]
            #         [ dx_deta dy_deta ]
            J_qp[i, j] = np.dot(dphi_dxi_qp[j].transpose(), x_n[i])
            
    return jnp.array(J_qp)


###################################################################################################
# Kernel 4


@timer(time_jit=True, n_calls=timer_n_calls)
def k4_global_to_param_jacobian_inv(J_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Kernel 4 - Calculate Jacobian matrices for global to parametric transformation at quad points

    Parameters
    ----------
    J_qp   : dense 4d-array with shape (N_ge, N_qp, N_xi, N_x)

    Returns
    -------
    G_qp   : dense 4d-array with shape (N_ge, N_qp, N_xi, N_x)
    """
    return jnp.linalg.inv(J_qp).transpose(0, 1, 3, 2)


@timer(time_jit=True, n_calls=timer_n_calls)
def k4_global_to_param_jacobian_inv_alt(J_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Alternative Kernel 4 - Calculate Jacobian matrices for global to parametric transformation at quad points

    Parameters
    ----------
    J_qp   : dense 4d-array with shape (N_ge, N_x, N_xi, N_qp)

    Returns
    -------
    G_qp   : dense 4d-array with shape (N_ge, N_x, N_xi, N_qp)
    """
    return jnp.linalg.inv(J_qp.transpose(0, 3, 2, 1)).transpose(0, 2, 3, 1)


@timer()
def k4_global_to_param_jacobian_loop(
    J_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert J_qp.shape[0] == dims.N_ge
    assert J_qp.shape[1] == dims.N_qp
    assert J_qp.shape[2] == dims.N_xi
    assert J_qp.shape[3] == dims.N_x

    G_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_xi, dims.N_x))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            G_qp[i, j] = np.linalg.inv(J_qp[i, j]).T
    return jnp.array(G_qp)


###################################################################################################
# Kernel 5


@timer(time_jit=True, n_calls=timer_n_calls)
def k5_calc_jacobian_det(J_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Kernel 5 - Calculate determinant of Jacobian matrices for global to parametric
    transformation at quad points.

    Parameters
    ----------
    J_qp   : dense 4d-array with shape (N_ge, N_qp, N_xi, N_x)

    Returns
    -------
    det_J_qp   : dense 2d-array with shape (N_ge, N_qp)
    """
    return jnp.linalg.det(J_qp)


@timer(time_jit=True, n_calls=timer_n_calls)
def k5_calc_jacobian_det_alt(J_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Alternative Kernel 5 - Calculate determinant of Jacobian matrices for global to parametric
    transformation at quad points.

    Parameters
    ----------
    J_qp   : dense 4d-array with shape (N_ge, N_x, N_xi, N_qp)

    Returns
    -------
    det_J_qp   : dense 2d-array with shape (N_ge, N_qp)
    """
    return jnp.linalg.det(J_qp.transpose(0, 3, 2, 1))


@timer()
def k5_calc_jacobian_det_loop(J_qp: jnp.ndarray, dims: Dimensions) -> jnp.ndarray:
    assert J_qp.shape[0] == dims.N_ge
    assert J_qp.shape[1] == dims.N_qp
    assert J_qp.shape[2] == dims.N_xi
    assert J_qp.shape[3] == dims.N_x

    det_J_qp = np.zeros((dims.N_ge, dims.N_qp))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            det_J_qp[i, j] = np.linalg.det(J_qp[i, j])
    return jnp.array(det_J_qp)


###################################################################################################
# Kernel 6


@timer(time_jit=True, n_calls=timer_n_calls)
def k6_basis_derivatives_global_einsum(
    G_qp: jnp.ndarray, dphi_dxi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 6 - Calculate basis function derivatives in global coordinate system at quad points.

    Parameters
    ----------
    G_qp   : dense 4d-array with shape (N_ge, N_qp, N_xi, N_x)
    dphi_dxi_qp : dense 3d-array with shape (N_qp, N_n, N_xi)

    Returns
    -------
    dphi_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_n, N_xi)
    """
    return jnp.einsum("ijkl,jmk->ijml", G_qp, dphi_dxi_qp)


@timer(time_jit=True, n_calls=timer_n_calls)
def k6_basis_derivatives_global_einsum_alt(
    G_qp: jnp.ndarray, dphi_dxi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Alternative Kernel 6 - Calculate basis function derivatives in global coordinate system at
    quad points.

    Parameters
    ----------
    G_qp   : dense 4d-array with shape (N_ge, N_x, N_xi, N_qp)
    dphi_dxi_qp : dense 3d-array with shape (N_n, N_xi, N_qp)

    Returns
    -------
    dphi_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_n, N_x)
    """
    return jnp.einsum("ijkl,mkl->ilmj", G_qp, dphi_dxi_qp)


@timer(time_jit=True, n_calls=timer_n_calls)
def k6_basis_derivatives_global_einsum_alt2(
    G_qp: jnp.ndarray, dphi_dxi_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Alternative Kernel 6 - Calculate basis function derivatives in global coordinate system at
    quad points.

    Parameters
    ----------
    G_qp   : dense 4d-array with shape (N_ge, N_qp, N_x, N_xi)
    dphi_dxi_qp : dense 3d-array with shape (N_qp, N_n, N_xi)

    Returns
    -------
    dphi_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_n, N_x)
    """
    return jnp.einsum("ijkl,jml->ijmk", G_qp, dphi_dxi_qp)


@timer()
def k6_basis_derivatives_global_loop(
    G_qp: jnp.ndarray, dphi_dxi_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert G_qp.shape[0] == dims.N_ge
    assert G_qp.shape[1] == dims.N_qp
    assert G_qp.shape[2] == dims.N_xi
    assert G_qp.shape[3] == dims.N_x
    assert dphi_dxi_qp.shape[0] == dims.N_qp
    assert dphi_dxi_qp.shape[1] == dims.N_n
    assert dphi_dxi_qp.shape[2] == dims.N_x

    dphi_dx_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_n, dims.N_x))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            dphi_dx_qp[i, j] = np.dot(dphi_dxi_qp[j], G_qp[i, j])
    return jnp.array(dphi_dx_qp)


###################################################################################################
# Kernel 7


@timer(time_jit=True, n_calls=timer_n_calls)
def k7_grad_solution_global_einsum(
    dphi_dx_qp: jnp.ndarray, u_n: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 7 - Calculate displacement gradient in global coordinate system at quad points.

    Parameters
    ----------
    dphi_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_n, N_x)
    u_n         : dense 4d-array with shape (N_ge, N_n, N_u)

    Returns
    -------
    du_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_u, N_x)
    """
    return jnp.einsum("ijkl,ikm->ijml", dphi_dx_qp, u_n)


@timer()
def k7_grad_solution_global_loop(
    dphi_dx_qp: jnp.ndarray, u_n: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert dphi_dx_qp.shape[0] == dims.N_ge
    assert dphi_dx_qp.shape[1] == dims.N_qp
    assert dphi_dx_qp.shape[2] == dims.N_n
    assert dphi_dx_qp.shape[3] == dims.N_x
    assert u_n.shape[0] == dims.N_ge
    assert u_n.shape[1] == dims.N_n
    assert u_n.shape[2] == dims.N_u

    du_dx_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_u, dims.N_x))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            du_dx_qp[i, j] = np.dot(
                dphi_dx_qp[i, j].transpose((1, 0)), u_n[i]
            ).transpose((1, 0))
    return jnp.array(du_dx_qp)


###################################################################################################
# Kernel 8


@timer(time_jit=True, n_calls=timer_n_calls)
def k8_strain(du_dx_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Kernel 8 - Calculate total strain in global coordinate system at quad points.

    Parameters
    ----------
    du_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_u, N_x)

    Returns
    -------
    eps_qp   : dense 4d-array with shape (N_ge, N_qp, N_x, N_x)
    """
    return 0.5 * (du_dx_qp + du_dx_qp.transpose((0, 1, 3, 2)))


@timer()
def k8_strain_loop(du_dx_qp: jnp.ndarray, dims: Dimensions) -> jnp.ndarray:
    assert du_dx_qp.shape[0] == dims.N_ge
    assert du_dx_qp.shape[1] == dims.N_qp
    assert du_dx_qp.shape[2] == dims.N_u
    assert du_dx_qp.shape[3] == dims.N_x

    eps_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_x, dims.N_x))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            eps_qp[i, j] = 0.5 * (du_dx_qp[i, j] + du_dx_qp[i, j].transpose())
    return jnp.array(eps_qp)


@timer(time_jit=True, n_calls=timer_n_calls)
def k8_strain_voigt(du_dx_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Kernel 8 - Calculate total strain in global coordinate system at quad points.

    Parameters
    ----------
    du_dx_qp   : dense 4d-array with shape (N_ge, N_qp, N_u, N_x)

    Returns
    -------
    eps_qp   : dense 3d-array with shape (N_ge, N_qp, N_eps)
    """
    if du_dx_qp.shape[2] == 1:  # 1D
        return 0.5 * (du_dx_qp + du_dx_qp.transpose((0, 1, 3, 2)))
    elif du_dx_qp.shape[2] == 2:  # 2D
        eps_qp = 0.5 * (du_dx_qp + du_dx_qp.transpose((0, 1, 3, 2)))
        # To voigt notation reshape
        return eps_qp[..., [0, 1, 0], [0, 1, 1]]
    elif du_dx_qp.shape[2] == 3:  # 3D
        eps_qp = 0.5 * (du_dx_qp + du_dx_qp.transpose((0, 1, 3, 2)))
        # To voigt notation reshape
        return eps_qp[..., [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    else:
        raise RuntimeError("Displacement must be 1D, 2D or 3D to compute strain.")


###################################################################################################
# Kernel 9


@timer(time_jit=True, n_calls=timer_n_calls)
def k9_stress_isotropic(mat_params_qp: jnp.ndarray, eps_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Kernel 9 - Calculate stress strain in global coordinate system at quad points.

    Parameters
    ----------
    mat_params_qp : dense 3d-array with shape (N_ge, N_qp, N_mp)
    eps_qp        : dense 4d-array with shape (N_ge, N_qp, N_x, N_x)

    Returns
    -------
    stress_qp   : dense 4d-array with shape (N_ge, N_qp, N_x, N_x)
    """
    # Material parameters should be E and nu
    assert mat_params_qp.shape[2] == 2
    E = mat_params_qp[..., 0]
    nu = mat_params_qp[..., 1]
    G = 0.5 * E / (1.0 + nu)
    zero = jnp.zeros_like(nu)

    if eps_qp.shape[2] == 1:  # 1D
        C_qp = E.transpose((1, 2, 0))
    elif eps_qp.shape[2] == 2:  # 2D
        # C_qp has shape (N_ge, N_qp, N_eps, N_eps)
        C_qp = jnp.linalg.inv(
            np.array(
                [
                    [1.0 / E, -nu / E, zero],
                    [-nu / E, 1.0 / E, zero],
                    [zero, zero, 1.0 / G],
                ]
            ).transpose((2, 3, 1, 0))
        )
    elif eps_qp.shape[2] == 3:  # 3D
        # C_qp has shape (N_ge, N_qp, N_eps, N_eps)
        C_qp = jnp.linalg.inv(
            np.array(
                [
                    [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                    [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                    [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                    [zero, zero, zero, 1.0 / G, zero, zero],
                    [zero, zero, zero, zero, 1.0 / G, zero],
                    [zero, zero, zero, zero, zero, 1.0 / G],
                ]
            ).transpose((2, 3, 1, 0))
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    # C: (N_ge, N_qp, N_eps, N_eps)
    # eps: (N_ge, N_qp, N_eps)
    return rank2_voigt_to_tensor(
        jnp.einsum("ijkl,ijl->ijk", C_qp, rank2_tensor_to_voigt(eps_qp))
    )


@timer()
def k9_stress_isotropic_loop(
    mat_params_qp: jnp.ndarray, eps_qp_tensor: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    """
    mat_params_qp : dense 3d-array with shape (N_ge, N_qp, N_mp)
    eps_qp        : dense 3d-array with shape (N_ge, N_qp, N_x, N_x)
    """
    assert mat_params_qp.shape[0] == dims.N_ge
    assert mat_params_qp.shape[1] == dims.N_qp
    assert mat_params_qp.shape[2] == dims.N_mp
    assert eps_qp_tensor.shape[0] == dims.N_ge
    assert eps_qp_tensor.shape[1] == dims.N_qp
    assert eps_qp_tensor.shape[2] == dims.N_x
    assert eps_qp_tensor.shape[3] == dims.N_x

    eps_qp = rank2_tensor_to_voigt(eps_qp_tensor)
    stress_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_eps))
    if eps_qp.shape[2] == 1:  # 1D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_qp[i, j, 0]
                stress_qp[i, j] = E * eps_qp[i, j]
    elif eps_qp.shape[2] == 3:  # 2D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_qp[i, j, 0]
                nu = mat_params_qp[i, j, 1]
                G = 0.5 * E / (1.0 + nu)
                zero = 0.0
                stress_qp[i, j] = np.dot(
                    jnp.linalg.inv(
                        np.array(
                            [
                                [1.0 / E, -nu / E, zero],
                                [-nu / E, 1.0 / E, zero],
                                [zero, zero, 1.0 / G],
                            ]
                        )
                    ),
                    eps_qp[i, j],
                )
    elif eps_qp.shape[2] == 6:  # 3D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_qp[i, j, 0]
                nu = mat_params_qp[i, j, 1]
                G = 0.5 * E / (1.0 + nu)
                zero = 0.0
                stress_qp[i, j] = np.dot(
                    jnp.linalg.inv(
                        np.array(
                            [
                                [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                                [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                                [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                                [zero, zero, zero, 1.0 / G, zero, zero],
                                [zero, zero, zero, zero, 1.0 / G, zero],
                                [zero, zero, zero, zero, zero, 1.0 / G],
                            ]
                        )
                    ),
                    eps_qp[i, j],
                )
    return rank2_voigt_to_tensor(jnp.array(stress_qp))


@timer(time_jit=True, n_calls=timer_n_calls)
def k9_stress_isotropic_voigt(
    mat_params_qp: jnp.ndarray, eps_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 9 - Calculate stress strain in global coordinate system at quad points.

    Parameters
    ----------
    mat_params_qp : dense 3d-array with shape (N_ge, N_qp, N_mp)
    eps_qp        : dense 3d-array with shape (N_ge, N_qp, N_eps)

    Returns
    -------
    stress_qp   : dense 4d-array with shape (N_ge, N_qp, N_eps)
    """
    # Material parameters should be E and nu
    assert mat_params_qp.shape[2] == 2
    E = mat_params_qp[..., 0]
    nu = mat_params_qp[..., 1]
    G = 0.5 * E / (1.0 + nu)
    zero = jnp.zeros_like(nu)

    if eps_qp.shape[2] == 1:  # 1D
        C_qp = E.transpose((1, 2, 0))
    elif eps_qp.shape[2] == 3:  # 2D
        # C_qp has shape (N_ge, N_qp, N_eps, N_eps)
        C_qp = jnp.linalg.inv(
            np.array(
                [
                    [1.0 / E, -nu / E, zero],
                    [-nu / E, 1.0 / E, zero],
                    [zero, zero, 1.0 / G],
                ]
            ).transpose((2, 3, 1, 0))
        )
    elif eps_qp.shape[2] == 6:  # 3D
        # C_qp has shape (N_ge, N_qp, N_eps, N_eps)
        C_qp = jnp.linalg.inv(
            np.array(
                [
                    [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                    [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                    [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                    [zero, zero, zero, 1.0 / G, zero, zero],
                    [zero, zero, zero, zero, 1.0 / G, zero],
                    [zero, zero, zero, zero, zero, 1.0 / G],
                ]
            ).transpose((2, 3, 1, 0))
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    # C: (N_ge, N_qp, N_eps, N_eps)
    # eps: (N_ge, N_qp, N_eps)
    return jnp.einsum("ijkl,ijl->ijk", C_qp, eps_qp)


@timer()
def k9_stress_isotropic_voigt_loop(
    mat_params_qp: jnp.ndarray, eps_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    """
    mat_params_qp : dense 3d-array with shape (N_ge, N_qp, N_mp)
    eps_qp        : dense 3d-array with shape (N_ge, N_qp, N_eps)
    """
    assert mat_params_qp.shape[0] == dims.N_ge
    assert mat_params_qp.shape[1] == dims.N_qp
    assert mat_params_qp.shape[2] == dims.N_mp
    assert eps_qp.shape[0] == dims.N_ge
    assert eps_qp.shape[1] == dims.N_qp
    assert eps_qp.shape[2] == dims.N_eps

    stress_qp = np.zeros((dims.N_ge, dims.N_qp, dims.N_eps))
    if eps_qp.shape[2] == 1:  # 1D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_qp[i, j, 0]
                stress_qp[i, j] = E * eps_qp[i, j]
    elif eps_qp.shape[2] == 3:  # 2D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_qp[i, j, 0]
                nu = mat_params_qp[i, j, 1]
                G = 0.5 * E / (1.0 + nu)
                zero = 0.0
                stress_qp[i, j] = np.dot(
                    jnp.linalg.inv(
                        np.array(
                            [
                                [1.0 / E, -nu / E, zero],
                                [-nu / E, 1.0 / E, zero],
                                [zero, zero, 1.0 / G],
                            ]
                        )
                    ),
                    eps_qp[i, j],
                )
    elif eps_qp.shape[2] == 6:  # 3D
        for i in range(dims.N_ge):
            for j in range(dims.N_qp):
                E = mat_params_qp[i, j, 0]
                nu = mat_params_qp[i, j, 1]
                G = 0.5 * E / (1.0 + nu)
                zero = 0.0
                stress_qp[i, j] = np.dot(
                    jnp.linalg.inv(
                        np.array(
                            [
                                [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                                [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                                [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                                [zero, zero, zero, 1.0 / G, zero, zero],
                                [zero, zero, zero, zero, 1.0 / G, zero],
                                [zero, zero, zero, zero, zero, 1.0 / G],
                            ]
                        )
                    ),
                    eps_qp[i, j],
                )
    return jnp.array(stress_qp)


###################################################################################################
# Kernel 10


@timer(time_jit=True, n_calls=timer_n_calls)
def k10_grad_dphi_dx_stress(
    dphi_dx_qp: jnp.ndarray, stress_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 10 - Calculates the product of the gradient of the derivative of the basis functions
    and the stress tensor (a measure of the force imbalance) at quad points.

    Essentially computes: dphi_i/dx_k * stress_kj -> result_ij but expressed in Voigt notation.
    dphi_i/dx_k * stress_kj =
        [ dphi_i/dx * stress_xj + dphi_i/dy * stress_yj + dphi_i/dz * stress_zj ]
    which has shape (N_n, N_x)

    Parameters
    ----------
    dphi_dx_qp    : dense 4d-array with shape (N_ge, N_qp, N_n, N_x)
    stress_qp     : dense 4d-array with shape (N_ge, N_qp, N_x, N_x)

    Returns
    -------
    grad_dphi_dx_stress_qp   : dense 4d-array with shape (N_ge, N_qp, N_n, N_x)
    """
    return jnp.einsum("ijkm,ijml->ijkl", dphi_dx_qp, stress_qp)


@timer()
def k10_grad_dphi_dx_stress_loop(
    dphi_dx_qp: jnp.ndarray, stress_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert dphi_dx_qp.shape[0] == dims.N_ge
    assert dphi_dx_qp.shape[1] == dims.N_qp
    assert dphi_dx_qp.shape[2] == dims.N_n
    assert dphi_dx_qp.shape[3] == dims.N_x
    assert stress_qp.shape[0] == dims.N_ge
    assert stress_qp.shape[1] == dims.N_qp
    assert stress_qp.shape[2] == dims.N_x
    assert stress_qp.shape[3] == dims.N_x

    grad_dphi_dx_stress = np.zeros((dims.N_ge, dims.N_qp, dims.N_n, dims.N_x))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            grad_dphi_dx_stress[i, j] = np.dot(dphi_dx_qp[i, j], stress_qp[i, j])
    return jnp.array(grad_dphi_dx_stress)


###################################################################################################
# Kernel 11


@timer(time_jit=True, n_calls=timer_n_calls)
def k11_residual(
    grad_dphi_dx_stress_qp: jnp.ndarray, det_J_qp: jnp.ndarray, W_qp: jnp.ndarray
) -> jnp.ndarray:
    """
    Kernel 11 - Calculate residual for element.

    Parameters
    ----------
    grad_dphi_dx_stress_qp : dense 4d-array with shape (N_ge, N_qp, N_n, N_x)
    det_J_qp               : dense 2d-array with shape (N_ge, N_qp)
    W_qp                   : dense 1d-array with shape (N_qp)

    Returns
    -------
    R_e  : dense 3d-array with shape (N_ge, N_n, N_x)
    """
    return jnp.einsum(
        "ijkl,ij->ikl", grad_dphi_dx_stress_qp, jnp.multiply(det_J_qp, W_qp)
    )


@timer()
def k11_residual_loop(
    grad_dphi_dx_stress_qp: jnp.ndarray, det_J_qp: jnp.ndarray, W_qp: jnp.ndarray, dims: Dimensions
) -> jnp.ndarray:
    assert grad_dphi_dx_stress_qp.shape[0] == dims.N_ge
    assert grad_dphi_dx_stress_qp.shape[1] == dims.N_qp
    assert grad_dphi_dx_stress_qp.shape[2] == dims.N_n
    assert grad_dphi_dx_stress_qp.shape[3] == dims.N_x
    assert det_J_qp.shape[0] == dims.N_ge
    assert det_J_qp.shape[1] == dims.N_qp
    assert W_qp.shape[0] == dims.N_qp

    R_e = np.zeros((dims.N_ge, dims.N_n, dims.N_u))
    for i in range(dims.N_ge):
        for j in range(dims.N_qp):
            R_e[i] += grad_dphi_dx_stress_qp[i, j] * det_J_qp[i, j] * W_qp[j]
    return jnp.array(R_e)
