from enum import Enum
from dataclasses import dataclass
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from ..basis_quadrature import FiniteElementType

class ContactBackend(Enum):
    JAX_HASH = "jax_hash"
    SCIPY_KDTREE = "scipy_kdtree"
    NEWTON_WARP = "newton_warp"
    JZTREE = "jztree"
    CUPYX_KDTREE = "cupyx_kdtree"
    AUTO = "auto"
    
def normalize_contact_backend(backend: ContactBackend | str) -> ContactBackend:
    if isinstance(backend, str):
        normalized = backend.lower().strip()
        if normalized in ("jaxhash", "jax_hash"):
            return ContactBackend.JAX_HASH
        if normalized in ("cupyx", "cupyx_kdtree"):
            return ContactBackend.CUPYX_KDTREE
        if normalized in ("scipy", "scipy_kdtree"):
            return ContactBackend.SCIPY_KDTREE
        if normalized in ("warp", "newton_warp"):
            return ContactBackend.NEWTON_WARP
        if normalized in ("jztree",):
            return ContactBackend.JZTREE
        if normalized in ("auto",):
            return ContactBackend.AUTO
        backend = ContactBackend(backend)
    if not isinstance(backend, ContactBackend):
        raise TypeError("backend must be a ContactBackend or contact backend string")
    return backend

def resolve_contact_backend(backend: ContactBackend | str, auto_uses_newton_warp: bool = False) -> ContactBackend:
    backend = normalize_contact_backend(backend)
    if backend == ContactBackend.AUTO:
        if auto_uses_newton_warp:
            from .warp_contact import NEWTON_WARP_AVAILABLE
            if NEWTON_WARP_AVAILABLE:
                return ContactBackend.NEWTON_WARP
        return ContactBackend.JAX_HASH
    if backend == ContactBackend.NEWTON_WARP:
        from .warp_contact import _require_newton_warp
        _require_newton_warp()
    elif backend == ContactBackend.JZTREE:
        from .jztree_contact import _require_jztree
        _require_jztree()
    elif backend == ContactBackend.CUPYX_KDTREE:
        from .cupyx_contact import _require_cupyx_kdtree
        _require_cupyx_kdtree()
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
    knn_k: int = 128
    hash_pad_size: int = 2
    cell_capacity_buffer: float = 1.5

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