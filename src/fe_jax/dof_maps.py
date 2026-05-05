from enum import Enum
from typing import List, Dict, Any
import copy

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class DofEnumeration:
    """
    Stores key information describing a degree of freedom (DoF) enumeration.
    """

    # The number of elements owned by this rank
    n_owned_elements: int = struct.field(pytree_node=False)

    # The number of DoF's owned by this rank (excludes ghosts).
    # Due to the global number scheme, this is equivalent to:
    # owned_global_dof_end - owned_global_dof_begin
    # Note: during the enrichment phase, this will be incorrect as new DoF's do not have global
    #       numbers until the renumbering phase.
    n_owned_dofs: int = struct.field(pytree_node=False)

    # The number of DoF's appearing in owned element DoF maps but are owned by other ranks.
    # In other words, this rank will probably write to containers for these DoF entries even
    # though this rank does not own them.
    n_local_ghost_dofs: int = struct.field(pytree_node=False)

    # The number of DoF's that ONLY appear in ghost elements and are owned by other ranks.
    # In other words, this rank will only read from containers for these DoF entries.
    n_exclusive_ghost_dofs: int = struct.field(pytree_node=False)

    # Specifies the number of DoFs that are not tied to any element / basis function.
    # Note: These DoFs will be owned by the last rank. Consequently, it will be included in the
    #       count of n_owned_dofs for the last rank, but will appear in the count of
    #       n_local_ghost_dofs for all other ranks (since they may write to entries corresponding
    #       to these DoFs).
    n_free_global_dofs: int = struct.field(pytree_node=False)

    # The rank index that marks the beginning of the free global DoFs.
    free_global_dof_rank_begin: int = struct.field(pytree_node=False)

    # Since global numbering scheme will maintain that the DoF's owned by this rank
    # lie within a continuous interval (with each rank's interval sorted by rank),
    # this is the beginning of the range (inclusive).
    # Note: during the enrichment phase, this will be incorrect as new DoF's
    #       do not have global numbers until the renumbering phase.
    owned_global_dof_begin: int = struct.field(pytree_node=False)

    # The end of the global owned range (inclusive).
    # Note: during the enrichment phase, this will be incorrect as new DoF's
    #       do not have global numbers until the renumbering phase.
    owned_global_dof_end: int = struct.field(pytree_node=False)

    # The map that permutes the rank-based DoF number to global number.
    rank_to_global_map: jnp.ndarray

    class DofType(Enum):
        OWNED = 0
        LOCAL_GHOST = 1
        EXCLUSIVE_GHOST = 2

    def n_dofs_on_rank(self) -> int:
        """
        The number of DoF's appearing in the DoF maps of all elements on this rank
        (owned and ghost).
        """
        return self.rank_to_global_map.shape[0]

    def n_local_dofs(self) -> int:
        """
        The number of DoF's appearing in the DoF maps of owned elements.
        This excludes DoF's that only appear in ghost elements.
        """
        return self.n_owned_dofs + self.n_local_ghost_dofs

    def get_dof_type(self, rank_dof: int) -> DofType:
        """
        Returns the type of DoF, which depends on which interval the DoF falls in.
        """
        if rank_dof < self.n_owned_dofs:
            return DofType.OWNED
        elif (
            self.n_owned_dofs <= rank_dof < self.n_owned_dofs + self.n_local_ghost_dofs
        ):
            return DofType.LOCAL_GHOST
        elif self.n_local_dofs() <= rank_dof < self.n_dofs_on_rank():
            return DofType.EXCLUSIVE_GHOST
        else:
            # Those outside of the ranges are assumed to be new DoF's during an enrichment phase,
            # which are owned.
            return DofType.OWNED
