import pytest
import numpy as np
import jax.numpy as jnp
from fe_jax.contact import (
    cupyx_contact_batch,
    scipy_contact_batch,
    raise_if_cupyx_kdtree_contact_capacity_exhausted,
    ContactCapacityError,
    CUPYX_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not CUPYX_AVAILABLE, reason="cupy/cupyx not available")


def test_cupyx_contact_batch_basic():
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)
    dummy_pair = np.array([0, 1], dtype=np.int32)

    capacity = 500
    cells, active, count, exhausted = cupyx_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.5,
        rigid_contact_max=capacity,
        dummy_pair=dummy_pair,
    )

    assert cells.shape == (capacity, 2)
    assert active.shape == (capacity,)
    assert int(count) > 0
    assert not bool(exhausted)
    assert int(jnp.sum(active)) == int(count)

    inactive = ~active
    assert np.all(np.asarray(cells[inactive]) == dummy_pair)


def test_cupyx_contact_batch_capacity_exhausted():
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)
    dummy_pair = np.array([0, 1], dtype=np.int32)

    capacity = 1
    cells, active, count, exhausted = cupyx_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.5,
        rigid_contact_max=capacity,
        dummy_pair=dummy_pair,
    )

    assert bool(exhausted)
    with pytest.raises(ContactCapacityError, match="rigid_contact_max=1"):
        raise_if_cupyx_kdtree_contact_capacity_exhausted(capacity, count, exhausted)


def test_cupyx_contact_matches_scipy():
    np.random.seed(42)
    N = 1000
    points = np.random.uniform(0, 5, size=(N, 3))
    fiber_ids = np.repeat(np.arange(10), 100)
    diameters = np.full(N, 0.5)

    scipy_cells = scipy_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.0,
    )

    capacity = 10000
    cells, active, count, exhausted = cupyx_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.0,
        rigid_contact_max=capacity,
    )

    assert not bool(exhausted)
    cp_active_cells = np.asarray(cells[active])

    def sort_contact(mat):
        mat = np.array(mat)
        mat = np.sort(mat, axis=1)
        idx = np.lexsort((mat[:, 1], mat[:, 0]))
        return mat[idx]

    s_sorted = sort_contact(scipy_cells)
    c_sorted = sort_contact(cp_active_cells)

    assert np.array_equal(s_sorted, c_sorted)
