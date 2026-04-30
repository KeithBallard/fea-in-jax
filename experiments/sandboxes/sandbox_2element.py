import jax
import jax.numpy as jnp
import numpy as np
from src.fe_jax import linear_elasticity
jax.config.update("jax_disable_jit", True)

u_Nd = jnp.array([[0,0],[0,-1732/150e3],[0,0]],dtype = jnp.float64)
x_Nd = jnp.array([[-5,0],[0,-5*np.sqrt(3)],[5,0]],dtype = jnp.float64)
cells = jnp.array([[0,1],[1,2]],dtype = jnp.int32)

F_const = u_Nd*0
F_stiff = u_Nd*0

dpx = jnp.array([[[-1],[1]],[[-1], [1]]])
W_q=jnp.array([0.5])
material_params = jnp.array([1e6,1])


def print_nice(U,X,R1,R2):
    d = U.shape[1]
    print('-'*(d*46+9))
    print(f"{'x_nd':^{d*6}} | {'u_nd':^{d*6}} | {'const':^{d*17}} | {'stiff':^{d*17}}")
    print('-'*(d*46+9))
    for u, x, r1, r2 in zip(U,X,R1,R2):
        # Format each coordinate and value to 6 decimal places
        x_str = "[" + " ".join(f"{xi:1.2f}" for xi in x) + "]"
        u_str = "[" + " ".join(f"{ui:1.2f}" for ui in u) + "]"
        r1_str = "[" + " ".join(f"{r1i: .8e}" for r1i in r1) + "]"
        r2_str = "[" + " ".join(f"{r2i: .8e}" for r2i in r2) + "]"
        print(f"{x_str:^{d*6}} | {u_str:^{d*6}} | {r1_str:^{d*17}} | {r2_str:^{d*17}}")

for c in cells:
    u_nd = u_Nd[c]
    x_nd = x_Nd[c]
    R_const = linear_elasticity.linear_truss_residual(
        u_nd=u_nd,
        x_nd=x_nd,
        dphi_dxi_qnp=dpx,
        W_q=W_q,
        material_params=material_params,
        constitutive_model = linear_elasticity.elastic_truss,
        internal_state_qi=[]
    )[0]
    R_stiff = linear_elasticity.stiffness_residual(
        u_nd=u_nd,
        x_nd=x_nd,
        material_params=material_params,
        internal_state_qi=[]
    )[0]
    F_const = F_const.at[c].add(R_const)
    F_stiff = F_stiff.at[c].add(R_stiff)
    # if print_test:
    #     print("\n Test %i:"%i)
    print_nice(u_nd,x_nd, R_const, R_stiff)

print(f"F_stiff = \n{F_stiff}\n")
print(f"F_const = \n{F_const}\n")
