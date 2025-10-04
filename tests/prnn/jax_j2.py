import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import NamedTuple

# Enable float64 precision if necessary
# jax.config.update("jax_enable_x64", True)

STATIC_MAX_ITER_YIELD = 25

class Material(NamedTuple):
    E: float
    nu: float
    sig0: float
    sigu: float
    b: float
    el_stiff: jnp.ndarray
    G: float
    K: float
    P: jnp.ndarray
    return_map_tol: float


class HistState(NamedTuple):
    eps_plastic: jnp.ndarray  # [n_points, 6]
    eps_p_eq: jnp.ndarray  # [n_points]


def create_material(E=3.13e3, nu=0.37, sig0=31.2, sigu=64.8, b=1/.003407):
    """Create material parameters."""
    el_stiff = jnp.zeros((6, 6))
    d = (1.0 + nu) * (1.0 - 2.0 * nu)

    # Fill diagonal terms
    el_stiff = el_stiff.at[0, 0].set(E * (1.0 - nu) / d)
    el_stiff = el_stiff.at[1, 1].set(E * (1.0 - nu) / d)
    el_stiff = el_stiff.at[2, 2].set(E * (1.0 - nu) / d)

    # Fill Poisson coupling terms
    poisson_val = E * nu / d
    el_stiff = el_stiff.at[0, 1].set(poisson_val)
    el_stiff = el_stiff.at[1, 0].set(poisson_val)
    el_stiff = el_stiff.at[0, 2].set(poisson_val)
    el_stiff = el_stiff.at[2, 0].set(poisson_val)
    el_stiff = el_stiff.at[1, 2].set(poisson_val)
    el_stiff = el_stiff.at[2, 1].set(poisson_val)

    # Fill shear terms
    shear_val = 0.5 * E / (1.0 + nu)
    el_stiff = el_stiff.at[3, 3].set(shear_val)
    el_stiff = el_stiff.at[4, 4].set(shear_val)
    el_stiff = el_stiff.at[5, 5].set(shear_val)

    K = E / 3.0 / (1.0 - 2.0 * nu)

    P = jnp.zeros((3, 3))
    P = P.at[0, 0].set(2. / 3.)
    P = P.at[1, 1].set(2. / 3.)
    P = P.at[0, 1].set(-1. / 3.)
    P = P.at[1, 0].set(-1. / 3.)
    P = P.at[2, 2].set(2.0)

    return Material(
        E=E,
        nu=nu,
        sig0=sig0,
        sigu=sigu,
        b=b,
        el_stiff=el_stiff,
        G=shear_val,
        K=K,
        P= P,
        return_map_tol=1e-7,
    )


def init_history(batch_size):
    """Initialize history variables for multiple integration points."""
    return HistState(
        eps_plastic=jnp.zeros((batch_size, 6)),
        eps_p_eq=jnp.zeros(batch_size)
    )


def sigma_C(x, material):
    """Yield stress as function of equivalent plastic strain."""
    return material.sig0 + (material.sigu - material.sig0) * (1 - jnp.exp(-material.b * x))


def sigma_C_deriv(x, material):
    """Derivative of yield stress function."""
    return (material.sigu - material.sig0) * material.b * jnp.exp(-material.b * x)


def t6_to_t3(t6):
    """Convert 6D stress/strain to 3D representation for plane stress."""
    return jnp.array([t6[0], t6[1], t6[3]]).reshape(3, 1)


def t3_to_t6(t3):
    """Convert 3D representation back to 6D."""
    t6 = jnp.zeros(6)
    t6 = t6.at[0].set(t3[0, 0])
    t6 = t6.at[1].set(t3[1, 0])
    t6 = t6.at[3].set(t3[2, 0])
    return t6


def ep3_to_ep6(ep3):
    """Convert 3D plastic strain to 6D."""
    ep6 = jnp.zeros(6)
    ep6 = ep6.at[0].set(ep3[0, 0])
    ep6 = ep6.at[1].set(ep3[1, 0])
    ep6 = ep6.at[2].set(-ep3[0, 0] - ep3[1, 0])  # out-of-plane component
    ep6 = ep6.at[3].set(ep3[2, 0])
    return ep6


def get_A_matrix(dgam, material):
    """Compute A matrix for stress update."""
    A_mat = jnp.zeros((3, 3))

    A11 = 3. * (1. - material.nu) / (3. * (1. - material.nu) + material.E * dgam)
    A22 = 1. / (1. + 2. * material.G * dgam)

    A1 = (A11 + A22) / 2
    A12 = (A11 - A22) / 2

    A_mat = A_mat.at[0, 0].set(A1)
    A_mat = A_mat.at[1, 0].set(A12)
    A_mat = A_mat.at[0, 1].set(A12)
    A_mat = A_mat.at[1, 1].set(A1)
    A_mat = A_mat.at[2, 2].set(A22)

    return A_mat


def eval_xi(dgam, A11_tr, A22_tr, A33_tr, material):
    """Evaluate xi and its derivative for the yield function."""
    fac = material.E / (1. - material.nu)
    f1 = 1. + fac * dgam / 3.
    f2 = 1. + 2. * material.G * dgam

    xi = A11_tr / (6 * f1 ** 2) + (0.5 * A22_tr + 2. * A33_tr) / f2 ** 2
    xi_der = -A11_tr * fac / (9 * f1 ** 3) - 2 * material.G * (A22_tr + 4 * A33_tr) / f2 ** 3

    return xi, xi_der


def eval_yield(dgam, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material):
    """Evaluate yield function value at dgam."""
    xi, _ = eval_xi(dgam, A11_tr, A22_tr, A33_tr, material)
    eps_p_eq = eps_p_eq_0 + dgam * jnp.sqrt(2. * xi / 3.)
    sigY = sigma_C(eps_p_eq, material)
    return 0.5 * xi - sigY ** 2 / 3., eps_p_eq


def eval_yield_and_der(dgam, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material):
    """Compute yield function value, derivative and plastic strain efficiently."""
    # Calculate xi and its derivative once
    xi, xi_der = eval_xi(dgam, A11_tr, A22_tr, A33_tr, material)

    # Calculate current equivalent plastic strain
    eps_p_eq = eps_p_eq_0 + dgam * jnp.sqrt(2. * xi / 3.)

    # Calculate yield stress and its derivative
    sigY = sigma_C(eps_p_eq, material)
    H = sigma_C_deriv(eps_p_eq, material)

    # Calculate yield function value
    yield_val = 0.5 * xi - sigY**2 / 3.

    # Calculate derivative terms
    sqrt_term = jnp.sqrt(2. / 3.) * (jnp.sqrt(xi) + dgam * xi_der / (2 * jnp.sqrt(xi)))
    H_bar = 2 * sigY * H * sqrt_term

    # Calculate yield derivative
    yield_der = xi_der / 2 - H_bar / 3.

    return yield_val, yield_der, eps_p_eq


def newton_step(dgam, old_dgam, old_yield_val, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material):
    """Single Newton-Raphson iteration step."""
    yield_val, yield_der, eps_p_eq = eval_yield_and_der(dgam, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material)

    ddgam = yield_val / yield_der

    # Divergence detection and handling
    ddgam = jax.lax.cond(
        (ddgam * old_dgam < 0) & (old_yield_val * yield_val < 0) & (jnp.abs(ddgam) > jnp.abs(old_dgam)),
        lambda _: -old_dgam * yield_val / (yield_val - old_yield_val),
        lambda _: ddgam,
        None
    )

    new_dgam = dgam - ddgam

    return new_dgam, ddgam, yield_val, eps_p_eq


def find_root_iter(carry, _):
    """Single iteration for find_root function."""
    dgam, old_ddgam, old_yield_val, i, converged, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material = carry

    # Perform Newton step
    new_dgam, ddgam, yield_val, eps_p_eq = newton_step(
        dgam, old_ddgam, old_yield_val, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material
    )

    # Check convergence
    new_converged = jnp.abs(yield_val) < material.return_map_tol

    return (new_dgam, ddgam, yield_val, i + 1, new_converged, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material), new_converged


def find_root(A11_tr, A22_tr, A33_tr, eps_p_eq_0, material):
    """Find root of yield function using Newton-Raphson method."""
    dgam_init = jnp.array(1e-10)
    old_ddgam_init = jnp.array(-1.0)

    xi_tr = A11_tr / 6.0 + 0.5 * A22_tr + 2.0 * A33_tr
    old_yield_val_init = 0.5 * xi_tr - sigma_C(eps_p_eq_0, material) ** 2 / 3.0

    # Initial state
    init_val = (
    dgam_init, old_ddgam_init, old_yield_val_init, 0, False, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material)

    # Run iterations
    final_state, _ = jax.lax.scan(
        find_root_iter,
        init_val,
        jnp.arange(STATIC_MAX_ITER_YIELD),
        length=STATIC_MAX_ITER_YIELD
    )

    dgam, _, _, iterations, converged, _, _, _, _, _ = final_state

    # Return dgam, eps_p_eq and convergence flag
    _, eps_p_eq = eval_yield(dgam, A11_tr, A22_tr, A33_tr, eps_p_eq_0, material)

    return dgam, eps_p_eq, converged


def update_single_point(eps_new, eps_p_hist, eps_p_eq_hist, material):
    """Update a single integration point."""

    # Calculate elastic strain
    eps_el = jnp.zeros(6)
    eps_el = eps_el.at[0].set(eps_new[0] - eps_p_hist[0])
    eps_el = eps_el.at[1].set(eps_new[1] - eps_p_hist[1])
    eps_el = eps_el.at[2].set(-material.nu / (1 - material.nu) * (eps_el[0] + eps_el[1]))
    eps_el = eps_el.at[3].set(eps_new[2] - eps_p_hist[3])

    # Calculate trial stress
    sig_tr = jnp.matmul(material.el_stiff, eps_el)

    # Compute values for plasticity check
    A11_tr = (sig_tr[0] + sig_tr[1]) ** 2
    A22_tr = (sig_tr[1] - sig_tr[0]) ** 2
    A33_tr = sig_tr[3] ** 2

    xi_tr = A11_tr / 6.0 + 0.5 * A22_tr + 2.0 * A33_tr
    f_tr = 0.5 * xi_tr - sigma_C(eps_p_eq_hist, material) ** 2 / 3.0

    # Check for plasticity
    is_plastic = f_tr >= material.return_map_tol

    # Function for plastic update
    def plastic_update():
        # Find root using Newton-Raphson
        dgam, new_eps_p_eq, _ = find_root(A11_tr, A22_tr, A33_tr, eps_p_eq_hist, material)
        # Convert to 3D representation for stress calculations
        sig_tr_3 = t6_to_t3(sig_tr)

        # Get A matrix and compute updated stress
        A_mat = get_A_matrix(dgam, material)
        stress_3 = jnp.matmul(A_mat, sig_tr_3)
        sig = t3_to_t6(stress_3)

        # Compute plastic strain increment
        depsp3 = dgam * jnp.matmul(material.P, stress_3)
        depsp = ep3_to_ep6(depsp3)

        # Update plastic strain
        new_eps_p = eps_p_hist + depsp

        # Return stress, new plastic strain and equivalent plastic strain
        stress = jnp.zeros_like(eps_new)
        stress = stress.at[0].set(sig[0])
        stress = stress.at[1].set(sig[1])
        stress = stress.at[2].set(sig[3])

        return stress, new_eps_p, new_eps_p_eq

    # Function for elastic update
    def elastic_update():
        stress = jnp.zeros_like(eps_new)
        stress = stress.at[0].set(sig_tr[0])
        stress = stress.at[1].set(sig_tr[1])
        stress = stress.at[2].set(sig_tr[3])

        return stress, eps_p_hist, eps_p_eq_hist

    # Choose between plastic and elastic update
    return jax.lax.cond(is_plastic, plastic_update, elastic_update)


# Vectorized constitutive update
update_vectorized = jit(vmap(update_single_point,
                             in_axes=(0, 0, 0, None),
                             out_axes=(0, 0, 0)))


def constitutive_update_batch(eps_new_batch, hist_state, material):
    """Update entire batch of integration points."""

    eps_plastic = hist_state[:, 0:6]
    eps_p_eq = hist_state[:, -1:].flatten()
    print('eps_new_batch', eps_new_batch.shape)
    print('eps_plastic', eps_plastic.shape)
    print('eps_p_eq', eps_p_eq.shape)

    stress_batch, new_eps_p_batch, new_eps_p_eq_batch = update_vectorized(
        eps_new_batch,
        eps_plastic,
        eps_p_eq,
        material
    )

    # Create new history state
    new_hist_state = HistState(
        eps_plastic=new_eps_p_batch,
        eps_p_eq=new_eps_p_eq_batch
    )

    return stress_batch, new_hist_state