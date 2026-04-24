"""
JAX-based constraint system for enforcing linear constraints on the solution vector.

See constraints.py for user-facing constraint definitions, which are then consolidated into a
ConstraintSystem for efficient enforcement.
"""

import jax.numpy as jnp
import jax
import jax.experimental.sparse as jsparse
import numpy as np

from typing import List
from flax import struct

from .constraints import FixedPointConstraint, MultiPointConstraint
from .utils import debug_print


@struct.dataclass
class ConstraintSystem:
    """
    Describes a linear constraint system of the form u_constrained = P u + g, where P is a sparse projection matrix
    and g is a vector of offsets.

    Nominally, P should be of the shape (# of reduced DOFs, # of all DOFs), where the reduced DOFs
    are the DOFs that are not constrained. This approach would result in entries for every DOF,
    even those that are not constrained. For efficiency, this stores the projection matrix and
    offset vector only for the DOFs that are not constrained.
    """

    # The DOFs that are constrained via linear constraints
    dep_dofs: jnp.ndarray

    # The projection matrix P (Sparse COO). Note: there should not be a non-zero entry for any
    # column corresponding to a dependent DOF. Otherwise, constrained DOFs could be functions of
    # other constrained DOFs. Instead, consolidation of constraints is required first.
    # Shape: (# of dependant DOF, # of all DOFs)
    P: jsparse.BCOO

    # The offset vector g
    # Shape: (# of dependant DOF,)
    g: jnp.ndarray

    @jax.jit
    def apply_to_solution(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the constraints to the solution vector.
        """
        return u.at[self.dep_dofs].set((self.P @ u) + self.g)

    @jax.jit
    def apply_to_delta_solution(self, delta_u: jnp.ndarray,u_0: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the constraints to the solution vector.
        """
        return delta_u.at[self.dep_dofs].set(self.P @ delta_u + self.g - u_0[self.dep_dofs])

    @jax.jit
    def apply_to_residual(self, R: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the constraints to the residual vector such that the "residual" for entries
        corresponding to dependent DOFs are the mismatch between the current solution and the
        value per the constraints.
        """
        return (
            (R + self.P.T @ R[self.dep_dofs])
            .at[self.dep_dofs]
            .set(u[self.dep_dofs] - self.g)
        )

    def __str__(self) -> str:
        n_constraints = self.dep_dofs.shape[0]
        n_total_dofs = self.P.shape[1]
        n_nnz = self.P.nse

        if n_total_dofs < 10 and n_constraints > 0:
            P_dense = self.P.todense()
            lines = []

            # Helper to format a row of a matrix
            def format_row(row):
                return "  ".join(f"{val:g}" for val in row)

            for i in range(n_constraints):
                dep_dof = self.dep_dofs[i]
                row_str = format_row(P_dense[i])
                g_val = f"{self.g[i]:g}"

                if i == 0:
                    prefix = f"[{dep_dof}] = "
                    suffix = f" [ u ] + [{g_val}]"
                    line = f"{prefix}[ {row_str} ]{suffix}"
                else:
                    # Align with the first line
                    prefix_0 = f"[{self.dep_dofs[0]}] = "
                    prefix_i = f"[{dep_dof}]"
                    padding = " " * (len(prefix_0) - len(prefix_i))

                    # Offset for [ u ] +
                    u_padding = " " * 9
                    line = f"{prefix_i}{padding}[ {row_str} ]{u_padding}[{g_val}]"

                lines.append(line)

            return "\n".join(lines)

        return (
            f"ConstraintSystem(\n"
            f"  n_constraints={n_constraints},\n"
            f"  n_total_dofs={n_total_dofs},\n"
            f"  n_nnz={n_nnz}\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()


def convert_constraints_to_system(
    fixed_point_constraints: List[FixedPointConstraint],
    multipoint_constraints: List[MultiPointConstraint],
    n_total_dofs: int,
) -> ConstraintSystem:
    """
    Converts lists of constraints (used for the consolidation step) to a ConstraintSystem (used
    for enforcement of constraints).
    """
    n_mpcs = len(multipoint_constraints)
    n_fpcs = len(fixed_point_constraints)
    n_constraints = n_mpcs + n_fpcs

    if n_constraints == 0:
        return ConstraintSystem(
            dep_dofs=jnp.array([], dtype=jnp.int32),
            P=jsparse.BCOO(
                (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)),
                shape=(0, n_total_dofs),
            ),
            g=jnp.array([], dtype=jnp.float32),
        )

    # First pass: count total non-zeros to pre-allocate arrays
    # Only MPCs contribute to P non-zeros (fixed point are P[row, :] = 0)
    nnz = sum(len(mpc.indep_dof_terms) for mpc in multipoint_constraints)

    # Allocate numpy arrays
    dep_dofs = np.empty(n_constraints, dtype=np.int32)
    g = np.empty(n_constraints, dtype=np.float32)

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)

    # Second pass: fill arrays
    current_nnz_idx = 0

    # Process MPCs
    for i, mpc in enumerate(multipoint_constraints):
        dep_dofs[i] = mpc.dep_dof
        g[i] = mpc.get_total_constant()

        for indep_dof, factor in mpc.indep_dof_terms.items():
            rows[current_nnz_idx] = i
            cols[current_nnz_idx] = indep_dof
            data[current_nnz_idx] = factor
            current_nnz_idx += 1

    # Process fixed point constraints
    for j, fpc in enumerate(fixed_point_constraints):
        idx = n_mpcs + j
        dep_dofs[idx] = fpc.dep_dof
        g[idx] = fpc.value
        # No P entries for fixed point constraints

    # Create JAX arrays from numpy arrays
    P_indices = (
        jnp.array(np.vstack((rows, cols)).T)
        if nnz > 0
        else jnp.zeros((0, 2), dtype=jnp.int32)
    )
    P_data = jnp.array(data) if nnz > 0 else jnp.zeros((0,), dtype=jnp.float32)

    P = jsparse.BCOO((P_data, P_indices), shape=(n_constraints, n_total_dofs))

    return ConstraintSystem(
        dep_dofs=jnp.array(dep_dofs, dtype=jnp.int32), P=P, g=jnp.array(g)
    )
