from enum import Enum
from dataclasses import dataclass
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from ..basis_quadrature import FiniteElementType

class ContactBackend(Enum):
    SCIPY_KDTREE = "scipy_kdtree"
    NEWTON_WARP = "newton_warp"
    AUTO = "auto"
    
def normalize_contact_backend(backend: ContactBackend | str) -> ContactBackend:
    if isinstance(backend,str):
        backend = ContactBackend(backend)
    if not isinstance(backend,ContactBackend):
        raise TypeError("backend must be a ContactBackend or contact backend string")
    return backend

def resolve_contact_backend(backend: ContactBackend | str, auto_uses_newton_warp: bool = True) -> ContactBackend:
    backend = normalize_contact_backend(backend)
    if backend == ContactBackend.AUTO:
        from .warp_contact import NEWTON_WARP_AVAILABLE
        return (
            ContactBackend.NEWTON_WARP
            if NEWTON_WARP_AVAILABLE and auto_uses_newton_warp
            else ContactBackend.SCIPY_KDTREE
        )
    if backend == ContactBackend.NEWTON_WARP:
        from .warp_contact import _require_newton_warp
        _require_newton_warp()
    return backend

class ContactCapacityError(OverflowError):
    pass

@dataclass
class ContactParams:
    self_adjacency_block: int
    contact_constitutive_model: jax.tree_util.Partial # such as `elastic_contact_truss_linear`

    D_stiffness_to_E_ratio: float # ratio between stiffness at the diameter distance to the stiffness of the truss elements (D/E) 
    M_stiffness_to_E_ratio: float # ratio between stiffness at M to the stiffness of the truss elements (M/E) 

    M_to_D_ratio: float # M is distance to start ramping up stiffness, so this is the ratio between M and the fiber diameter (M/D)
    C_to_D_ratio: float # C is distance to have a hard stiffness set.
    # It should be C_to_D_ratio<M_to_D_ratio<contact_search_alpha
    contact_search_alpha: float # dimensionless value for search_radius = contact_search_alpha*(radius1+radius2)
    contact_backend: ContactBackend = ContactBackend.AUTO
    rigid_contact_max: int | None = None

@struct.dataclass
class ContactMaterialSpec:
    E_c: float
    area: float
    M_to_D_ratio: float
    C_to_D_ratio: float
    search_alpha: float
    E_min: float

@dataclass
class ContactPreprocessConfig:
    vertices_fiber_ids: np.ndarray
    radius: float
    self_adjacency_block: int
    material_params: jnp.ndarray
    fe_type: FiniteElementType
    constitutive_model: jax.tree_util.Partial
    contact_pair_generator: jax.tree_util.Partial