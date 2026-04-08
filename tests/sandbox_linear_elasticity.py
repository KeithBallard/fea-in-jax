import jax
import jax.numpy as jnp
import numpy as np
from src.fe_jax import linear_elasticity
jax.config.update("jax_disable_jit", True)

# u_nd = jnp.array([[0.1,0.05,0.025],[0.01,0.2,0.03]])
# x_nd = jnp.array([[0,0.2,1],[0.25,0,1.1]])
dpx = jnp.array([[[-1],[1]],[[-1], [1]]])
W_q=jnp.array([0.5])
material_params = jnp.array([1e9,1])

for i in range(20):
    u_nd = jnp.array(np.random.rand(2)).reshape((2,1))
    x_nd = jnp.array(np.random.rand(2)).reshape((2,1))
    print(i,u_nd,x_nd)
    R_const = linear_elasticity.linear_truss_residual(u_nd=u_nd, x_nd=x_nd, dphi_dxi_qnp=dpx, W_q=W_q, material_params=material_params,constitutive_model = linear_elasticity.elastic_truss,internal_state_qi=[])[0]
    R_stiff = linear_elasticity.stiffness_residual(u_nd=u_nd, x_nd=x_nd,material_params=material_params,internal_state_qi=[])[0]
    
    assert jnp.isclose(R_const,R_stiff).all(), "residuals form different methods do not match"
print("\n For all randomly generated u_nd and x_nd in R^1 the residuals always matched.")

for i in range(20):
    u_nd = jnp.array(np.random.rand(4)).reshape((2,2))
    x_nd = jnp.array(np.random.rand(4)).reshape((2,2))
    print(i,u_nd,x_nd)
    R_const = linear_elasticity.linear_truss_residual(u_nd=u_nd, x_nd=x_nd, dphi_dxi_qnp=dpx, W_q=W_q, material_params=material_params,constitutive_model = linear_elasticity.elastic_truss,internal_state_qi=[])[0]
    R_stiff = linear_elasticity.stiffness_residual(u_nd=u_nd, x_nd=x_nd,material_params=material_params,internal_state_qi=[])[0]
    
    assert jnp.isclose(R_const,R_stiff).all(), "residuals form different methods do not match"
print("\n For all randomly generated u_nd and x_nd in R^2 the residuals always matched.")

for i in range(20):
    u_nd = jnp.array(np.random.rand(6)).reshape((2,3))
    x_nd = jnp.array(np.random.rand(6)).reshape((2,3))
    print(i,u_nd,x_nd)
    R_const = linear_elasticity.linear_truss_residual(u_nd=u_nd, x_nd=x_nd, dphi_dxi_qnp=dpx, W_q=W_q, material_params=material_params,constitutive_model = linear_elasticity.elastic_truss,internal_state_qi=[])[0]
    R_stiff = linear_elasticity.stiffness_residual(u_nd=u_nd, x_nd=x_nd,material_params=material_params,internal_state_qi=[])[0]
    
    assert jnp.isclose(R_const,R_stiff).all(), "residuals form different methods do not match"
print("\n For all randomly generated u_nd and x_nd in R^3 the residuals always matched.")
