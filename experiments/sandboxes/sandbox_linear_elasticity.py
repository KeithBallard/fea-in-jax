import jax
import jax.numpy as jnp
import numpy as np
from src.fe_jax import linear_elasticity
jax.config.update("jax_disable_jit", True)

dpx = jnp.array([[[-1],[1]],[[-1], [1]]])
W_q=jnp.array([0.5])
material_params = jnp.array([1e9,1])


def print_nice(U,X,R1,R2):
    d = U.shape[1]
    print('-'*(d*34+9))
    print(f"{'x_nd':^{d*6}} | {'u_nd':^{d*6}} | {'const':^{d*11}} | {'stiff':^{d*11}}")
    print('-'*(d*34+9))
    for u, x, r1, r2 in zip(U,X,R1,R2):
        # Format each coordinate and value to 6 decimal places
        x_str = "[" + " ".join(f"{xi:1.2f}" for xi in x) + "]"
        u_str = "[" + " ".join(f"{ui:1.2f}" for ui in u) + "]"
        r1_str = "[" + " ".join(f"{r1i: .2e}" for r1i in r1) + "]"
        r2_str = "[" + " ".join(f"{r2i: .2e}" for r2i in r2) + "]"
        print(f"{x_str:^{d*6}} | {u_str:^{d*6}} | {r1_str:^{d*11}} | {r2_str:^{d*11}}")

def compare_residuals(num_tests,print_test=False):
    for n in range(1,4):
        for i in range(num_tests):
            u_nd = jnp.array(np.random.rand(2*n)).reshape((2,n))
            x_nd = jnp.array(np.random.rand(2*n)).reshape((2,n))
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
                internal_state_qi=[],
                dphi_dxi_qnp=None,
                W_q=None,
                constitutive_model=None
            )[0]
            if print_test:
                print("\n Test %i:"%i)
                print_nice(u_nd,x_nd, R_const, R_stiff)
            assert jnp.isclose(R_const,R_stiff).all(), "residuals from different methods do not match"
        pass_str = f"* For all ({num_tests}) randomly generated u_nd and x_nd in R^{n} the residuals always matched. *"
        border = '*'*len(pass_str)
        print(f"\n{border}\n{pass_str}\n{border}")

compare_residuals(20,print_test=False)
