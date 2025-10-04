from helper import *


@partial(jax.jit, static_argnames=["D"])
def linear_isotropic(eps_eqdd: jnp.ndarray, material_params_eqm: jnp.ndarray, D: int):

    E = material_params_eqm[..., 0]
    nu = material_params_eqm[..., 1]
    G = 0.5 * E / (1.0 + nu)
    zero = jnp.zeros_like(nu)
    if D == 1:  # 1D
        C_eqss = E.transpose((1, 2, 0))[:, jnp.newaxis]
    elif D == 2:  # 2D
        C_eqss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, zero],
                    [-nu / E, 1.0 / E, zero],
                    [zero, zero, 1.0 / G],
                ]
            ).transpose((2, 3, 1, 0))
        )
    elif D == 3:  # 3D
        C_eqss = jnp.linalg.inv(
            jnp.array(
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

    stress_eqdd = rank2_voigt_to_tensor(
        jnp.einsum("ijkl,ijl->ijk", C_eqss, rank2_tensor_to_voigt(eps_eqdd))
    )
    return stress_eqdd


@partial(jax.jit, static_argnames=["D"])
def linear_isotropic_local(
    eps_qdd: jnp.ndarray, material_params_qm: jnp.ndarray, D: int
):

    E = material_params_qm[..., 0]
    nu = material_params_qm[..., 1]
    G = 0.5 * E / (1.0 + nu)
    zero = jnp.zeros_like(nu)
    if D == 1:  # 1D
        C_qss = E.transpose((1, 0))[:, jnp.newaxis]
    elif D == 2:  # 2D
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, zero],
                    [-nu / E, 1.0 / E, zero],
                    [zero, zero, 1.0 / G],
                ]
            ).transpose((2, 0, 1))
        )
    elif D == 3:  # 3D
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

    stress_qdd = rank2_voigt_to_tensor(
        jnp.einsum("qsi,qi->qs", C_qss, rank2_tensor_to_voigt(eps_qdd))
    )
    return stress_qdd


def _element_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_qm: jnp.ndarray,
):

    D = u_nd.shape[1]
    P = dphi_dxi_qnp.shape[2]

    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)
    assert (
        P == D
    )  # Next line requires this, otherwise a different approach is needed (i.e. shell elements)
    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    du_dx_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd)
    eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)))
    stress_qdd = linear_isotropic_local(eps_qdd, material_params_qm, D=D)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd


@jax.jit
def calculate_residual_batch_element_kernel(
    u_end: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_eqm: jnp.ndarray,
):
    """
    Calculates the element-level residual vectors for a batch of elements without any modification
    of the solution or residual to accomodate Dirichlet constraints. Called by calculate_residual.

    TODO document parameters
    """
    E = u_end.shape[0]
    N = u_end.shape[1]
    D = u_end.shape[2]
    Q = dphi_dxi_qnp.shape[0]
    P = dphi_dxi_qnp.shape[2]
    M = material_params_eqm.shape[2]

    J_eqpd = jnp.einsum("end,qnp->eqpd", x_end, dphi_dxi_qnp)
    assert (
        P == D
    )  # Next line requires this, otherwise a different approach is needed (i.e. shell elements)
    G_eqpd = jnp.linalg.inv(J_eqpd).transpose(0, 1, 3, 2)
    det_J_eq = jnp.linalg.det(J_eqpd)
    dphi_dx_eqnd = jnp.einsum("eqpd,qnp->eqnd", G_eqpd, dphi_dxi_qnp)
    du_dx_eqdd = jnp.einsum("eqnd,eni->eqid", dphi_dx_eqnd, u_end)
    eps_eqdd = 0.5 * (du_dx_eqdd + du_dx_eqdd.transpose((0, 1, 3, 2)))
    stress_eqdd = linear_isotropic(eps_eqdd, material_params_eqm, D=D)

    grad_dphi_dx_stress_eqnd = jnp.einsum("eqni,eqid->eqnd", dphi_dx_eqnd, stress_eqdd)
    det_JxW_eq = jnp.einsum("eq,q->eq", det_J_eq, W_q)
    R_end = jnp.einsum("eqnd,eq->end", grad_dphi_dx_stress_eqnd, det_JxW_eq)

    return R_end


@jax.jit
def calculate_residual_batch_element_kernel_vmap(
    u_end: jnp.ndarray,
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_eqm: jnp.ndarray,
):
    """
    Calculates the element-level residual vectors for a batch of elements without any modification
    of the solution or residual to accomodate Dirichlet constraints. Called by calculate_residual.

    TODO document parameters
    """

    @jax.vmap
    def residual_kernel(u_nd, x_nd, material_params_qm):
        return _element_residual(u_nd, x_nd, dphi_dxi_qnp, W_q, material_params_qm)

    R_end = residual_kernel(u_end, x_end, material_params_eqm)

    return R_end

u_end = jnp.array([[[0., 0.], [1., 0.], [0., 1.]]])
x_end = jnp.array([[[0., 0.], [1., 0.], [0., 1.]]])
fe_type = FiniteElementType(
    cell_type=CellType.triangle,
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=3,
)
xi_qp, W_q = get_quadrature(fe_type=fe_type)
phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)
material_params_eqm = jnp.zeros((1, xi_qp.shape[0], 2))
material_params_eqm = material_params_eqm.at[:,:,0].set(1e9)
material_params_eqm = material_params_eqm.at[:,:,1].set(0.3)

R_1 = calculate_residual_batch_element_kernel(u_end, x_end, dphi_dxi_qnp, W_q, material_params_eqm)
R_2 = calculate_residual_batch_element_kernel_vmap(u_end, x_end, dphi_dxi_qnp, W_q, material_params_eqm)

print(R_1)
print(R_2) 