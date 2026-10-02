import pytest
import numpy as np
import jax.numpy as jnp
from fe_jax.contact import (
    jztree_contact_batch,
    scipy_contact_batch,
    raise_if_jztree_contact_capacity_exhausted,
    ContactCapacityError,
    JZTREE_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not JZTREE_AVAILABLE, reason="jztree not available")


def test_jztree_contact_batch_basic():
    # Two fibers, 100 points each, placed close enough to contact
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)
    dummy_pair = np.array([0, 1], dtype=np.int32)

    capacity = 500
    cells, active, count, exhausted = jztree_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.5,
        rigid_contact_max=capacity,
        dummy_pair=dummy_pair,
        k=16,
    )

    assert cells.shape == (capacity, 2)
    assert active.shape == (capacity,)
    assert int(count) > 0
    assert not bool(exhausted)
    assert int(jnp.sum(active)) == int(count)

    # Inactive slots should be filled with dummy_pair
    inactive = ~active
    assert np.all(np.asarray(cells[inactive]) == dummy_pair)


def test_jztree_contact_batch_capacity_exhausted():
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)
    dummy_pair = np.array([0, 1], dtype=np.int32)

    # Deliberately undersized capacity
    capacity = 1
    cells, active, count, exhausted = jztree_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.5,
        rigid_contact_max=capacity,
        dummy_pair=dummy_pair,
        k=16,
    )

    assert bool(exhausted)
    with pytest.raises(ContactCapacityError, match="rigid_contact_max=1"):
        raise_if_jztree_contact_capacity_exhausted(capacity, count, exhausted)


def test_jztree_contact_matches_scipy():
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

    capacity = 5000
    cells, active, count, exhausted = jztree_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.0,
        rigid_contact_max=capacity,
        k=32,
    )

    assert not bool(exhausted)
    jz_active_cells = np.asarray(cells[active])

    def sort_contact(mat):
        mat = np.array(mat)
        mat = np.sort(mat, axis=1)
        idx = np.lexsort((mat[:, 1], mat[:, 0]))
        return mat[idx]

    s_sorted = sort_contact(scipy_cells)
    j_sorted = sort_contact(jz_active_cells)

    assert np.array_equal(s_sorted, j_sorted)
