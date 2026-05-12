import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from fe_jax.constraint_system import ConstraintSystem

# Create the user's example system
# [0] = 0.1 * [2]
# [1] = 0.2 * [2] + 0.1
# Total DOFs = 3
dep_dofs = jnp.array([0, 1], dtype=jnp.int32)
P_data = jnp.array([0.1, 0.2], dtype=jnp.float32)
P_indices = jnp.array([[0, 2], [1, 2]], dtype=jnp.int32)
P = jsparse.BCOO((P_data, P_indices), shape=(2, 3))
g = jnp.array([0.0, 0.1], dtype=jnp.float32)

cs = ConstraintSystem(dep_dofs=dep_dofs, P=P, g=g)
print("Printing ConstraintSystem (Small):")
print(cs)

# Test repr
print("\nRepr of ConstraintSystem:")
print(repr(cs))
