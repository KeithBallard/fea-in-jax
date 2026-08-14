import jax
import jax.numpy as jnp

def calc_I1(eps_d):
    '''
    calculate  first invariant of the stress 
    input: s11,s22,s12 are the stress tensors
    
    return: I1 stress
    '''
    e11 = eps_d[0]
    e22 = eps_d[1]
    I1 = e11 + e22
    return I1

def calc_J2(eps_d):
    '''
    calculate second invariant of the deviatoric stress 
    input: s11,s22,s12 are the stress tensors
    ouput: J2 stress
    '''
    e11 = eps_d[0]
    e22 = eps_d[1]
    e12 = eps_d[2]

    return ((e11-e22)**2 +e11**2 + e22**2) / 6 + e12**2

def calc_dmg_crit(eps_d,material_params_m):
    '''
    calculate yield criterion  
    input: 
        J2: second invariant of the deviatoric stress 
        I1: first invariant of the stress 
        sigmay_c: dmg strengths of the epoxy under compression
        sigmay_t: dmg strengths of the epoxy under tension
    ouput: yield_criterion
    '''        
    eps_c = material_params_m[5]
    eps_t = material_params_m[6]

    I1 = calc_I1(eps_d)
    J2 = calc_J2(eps_d)
    dmg_crit = 6*J2 + 2*I1*(eps_c-eps_t) - 2*eps_c*eps_t
    return dmg_crit


def calc_yield_crit(stress_d,material_params_m):
    '''
    calculate yield criterion  
    input: 
        J2: second invariant of the deviatoric stress 
        I1: first invariant of the stress 
        sigmay_c: yield strengths of the epoxy under compression
        sigmay_t: yield strengths of the epoxy under tension
    ouput: yield_criterion
    '''        
    sig_c = material_params_m[7]
    sig_t = material_params_m[8]

    I1 = calc_I1(stress_d)
    J2 = calc_J2(stress_d)
    yield_crit = 6*J2 + 2*I1*(sig_c-sig_t) - 2*sig_c*sig_t
    return yield_crit


def calc_d(Y,d,tau0,dt,material_params_m, eps_cd,C_ss):
    '''
    material_params_qm: jnp.ndarray,
    prev_strain: jnp.ndarray,
    '''
    # Constants -  Material properties
    E    = material_params_m[0]
    nu   = material_params_m[1]
    A    = material_params_m[2]
    B    = material_params_m[3]
    mu   = material_params_m[4]

    # D_mat = compute_C_qss_matrix(E,nu)
    # Compute the Strain energy
    DE = jnp.dot(C_ss, eps_cd)
    strain_energy = 0.5 * jnp.dot(eps_cd, DE)
    
    # damage threshold
    tau = jnp.sqrt(2 * strain_energy)

    update_tau0 = jax.lax.cond(tau0 == 0,
                        lambda x: tau.copy()*0.999,
                        lambda x: tau0,
                        operand=None)
    
    # Compute damage function, G
    G = 1 - update_tau0/tau*(1-A) - A*jnp.exp(B*(update_tau0-tau))

    # Compute current damage, d
    update_d = jax.lax.cond(G-Y >= 0,
                    lambda x: d + (mu * dt / (1 + mu * dt)) * (G - Y),
                    lambda x: d,
                    operand=None)

    # Compute current damage thresh Y
    curr_damage_thresh = (Y + mu*dt*G) / (1 + mu*dt)
    update_Y = jax.lax.cond(curr_damage_thresh > Y,
                    lambda x: curr_damage_thresh,
                    lambda x: Y,
                    operand=None)

    return update_Y, update_d, update_tau0

def calc_von_Mises(stress_cd):
    '''
    calculate von mises stress 
    input: s11,s22,s12 are the stress tensors
    ouput: von mises stress
    '''    
    s11 = stress_cd[0]
    s22 = stress_cd[1]
    s12 = stress_cd[2]
    return jnp.sqrt(s11**2 + s22**2 - s11 * s22 + 3.0 * s12**2)


def calc_H(von_mises,initial_VM,material_params_m):
    '''
    calculate slope of the tangent line to the hardening portion of the material’s constitutive behavior
    input: 
        H_ro, n_ro: Epoxy hardening properties
        initial_VM
        von_mises
    ouput: H
    '''   
    H_ro = material_params_m[9]
    n_ro = material_params_m[10]

    return H_ro * (initial_VM/von_mises)**n_ro
    
def calc_dfdsigma(stress_cd, von_Mises):
    '''
    calculate gradient of the von Mises yield function
    input: s11,s22,s12 are the stress tensors
    ouput: dfdsigma
    '''        
    s11 = stress_cd[0]
    s22 = stress_cd[1]
    s12 = stress_cd[2]
    dfdsigma = jnp.array([1/(2*von_Mises)*(2*s11-s22),
                          1/(2*von_Mises)*(2*s22-s11),
                          1/(2*von_Mises)*(6*s12)
                         ])
    return dfdsigma

def calc_temp_tensor(dfdsigma):
    '''
    outer product (or tensor product) of the vector dfdsigma
    '''
    return jnp.outer(dfdsigma, dfdsigma)

# def calc_temp_vector(dfdsigma, Dmat):
#     return jnp.dot(Dmat, dfdsigma)

def calc_temp_scalar(dfdsigma, Dmat):
    # return jnp.dot(dfdsigma, temp_vector)
    return jnp.dot(dfdsigma, jnp.dot(Dmat, dfdsigma))

def calc_Dmat_bar_increment(temp_tensor, Dmat_bar, H, temp_scalar):
    Dmat_bar_increment = jnp.dot(jnp.dot(Dmat_bar, temp_tensor), Dmat_bar)
    correction = Dmat_bar_increment * (-1.0 / (H + temp_scalar))
    return correction

def calc_update_Dmat(material_params_qm, vM0, stress_cd,C_ss):
    # Initalize
    # Calculate von Mises stress
    von_Mises = calc_von_Mises(stress_cd)

    # Update initial hardening von Mises
    vM0 = jax.lax.cond(vM0 == 0,
                        lambda x: von_Mises*0.999,
                        lambda x: vM0,
                        operand=None)

    # slope of the tangent line to the hardening portion of the material’s constitutive behavior
    H = calc_H(von_Mises,vM0,material_params_qm)
    
    # gradient of the von Mises yield function
    df_dsigma = calc_dfdsigma(stress_cd,von_Mises)

    # Calculate increment
    temp_tensor = calc_temp_tensor(df_dsigma)
    temp_scalar = calc_temp_scalar(df_dsigma, C_ss)

    C_ss_increment = calc_Dmat_bar_increment(temp_tensor,C_ss, H, temp_scalar)

    return vM0, C_ss_increment