from .boundary_conditions import *
from .constraints import *

import jax
import jax.numpy as jnp

from flax import struct

@dataclass
class NeumannCondition:
    """
    Represents a force of value applied at DoF.
    """

    dep_dof: int
    value: float


def convert_boundary_conditions_to_external_load(
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC],
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    dof_enumeration: DofEnumeration,
    n_solution_components: int,
    global_values: List[int] | None = None,
):
    """
    Searches the list of boundary conditions and converts the Neumann
    conditions to data type NeumannCondition.
    """
    external_load = []
    for bc in boundary_conditions:
        if isinstance(bc, NeumannBC):
            if bc.bc_type == BCType.NODE:
                external_load.append(
                    NeumannCondition(
                        dep_dof=n_solution_components * bc.index + bc.component,
                        value=bc.value,
                    )
                )
    return external_load


@struct.dataclass
class LoadSystem:
    dep_dofs: jnp.ndarray
    loads: jnp.ndarray

    @jax.jit
    def apply_to_residual(self, R: jnp.ndarray):
        return R.at[self.dep_dofs].set((R[self.dep_dofs] - self.loads))


def convert_external_load_to_system(
    external_load,
):
    n_loads = len(external_load)
    if n_loads == 0:
        return LoadSystem(
            dep_dofs=jnp.array([], dtype=jnp.int32),
            loads=jnp.array([], dtype=jnp.float32),
        )

    dep_dofs = np.empty(n_loads, dtype=np.int32)
    loads = np.empty(n_loads, dtype=np.float32)

    for i, el in enumerate(external_load):
        dep_dofs[i] = el.dep_dof
        loads[i] = el.value
    return LoadSystem(
        dep_dofs=jnp.array(dep_dofs, dtype=jnp.int32),
        loads=jnp.array(loads, dtype=jnp.float32),
    )