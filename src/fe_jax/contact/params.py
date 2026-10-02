import jax
import jax.numpy as jnp
import numpy as np
from .contact_types import ContactMaterialSpec

CONTACT_E_C_PARAM = 0
CONTACT_AREA_PARAM = 1
CONTACT_RADIUS_0_PARAM = 2
CONTACT_RADIUS_1_PARAM = 3
CONTACT_M_TO_D_PARAM = 4
CONTACT_C_TO_D_PARAM = 5
CONTACT_SEARCH_ALPHA_PARAM = 6
CONTACT_E_MIN_PARAM = 7
CONTACT_ACTIVE_PARAM = 8
CONTACT_MATERIAL_PARAM_COUNT = 9

def build_contact_material_params(
    contact_cells: jnp.ndarray,
    point_radii: jnp.ndarray,
    spec: ContactMaterialSpec,
    active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    contact_cells = jnp.asarray(contact_cells, dtype=jnp.int32)
    point_radii = jnp.asarray(point_radii)

    n_contact = contact_cells.shape[0]
    dtype = point_radii.dtype

    if active is None:
        active = jnp.ones((n_contact,), dtype=dtype)
    else:
        active = jnp.asarray(active, dtype=dtype)

    params = jnp.zeros((n_contact, CONTACT_MATERIAL_PARAM_COUNT), dtype=dtype)
    params = params.at[:, CONTACT_E_C_PARAM].set(spec.E_c)
    params = params.at[:, CONTACT_AREA_PARAM].set(spec.area)
    params = params.at[:, CONTACT_RADIUS_0_PARAM].set(point_radii[contact_cells[:, 0]])
    params = params.at[:, CONTACT_RADIUS_1_PARAM].set(point_radii[contact_cells[:, 1]])
    params = params.at[:, CONTACT_M_TO_D_PARAM].set(spec.M_to_D_ratio)
    params = params.at[:, CONTACT_C_TO_D_PARAM].set(spec.C_to_D_ratio)
    params = params.at[:, CONTACT_SEARCH_ALPHA_PARAM].set(spec.search_alpha)
    params = params.at[:, CONTACT_E_MIN_PARAM].set(spec.E_min)
    params = params.at[:, CONTACT_ACTIVE_PARAM].set(active)

    return params

@jax.jit
def pack_fixed_contact_cells(
    node0: jnp.ndarray,
    node1: jnp.ndarray,
    active: jnp.ndarray,
    count: jnp.ndarray,
    capacity: int,
    dummy_pair: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    capacity_exhausted = count >= capacity

    contact_cells = jnp.stack([node0,node1],axis=1).astype(jnp.int32)
    dummy_pair = jnp.asarray(dummy_pair, dtype=contact_cells.dtype)
    contact_cells = jnp.where(active[:, None], contact_cells, dummy_pair[None, :])

    return contact_cells, active, count, capacity_exhausted
 
def find_nonzero_length_dummy_contact_pair(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    for i in range(points.shape[0]):
        for j in range(i+1, points.shape[0]):
            if np.linalg.norm(points[j] - points[i])>0.0:
                return np.array([i,j],dtype=np.int32)
    raise ValueError("fixed-capacity contact requires at least one nonzero-length dummy pair")