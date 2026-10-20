import pytest
import numpy as np
import jax
import jax.numpy as jnp
from fe_jax.contact import (
    jaxhash_contact_batch,
    pad_hash_cells,
    raise_if_jaxhash_contact_capacity_exhausted,
    ContactCapacityError,
    ContactBackend,
    resolve_contact_backend,
    normalize_contact_backend,
    scipy_contact_batch,
    get_cuda_hardware_info,
    compute_dynamic_batch_params,
)


def test_jaxhash_default_backend_resolution():
    assert resolve_contact_backend(ContactBackend.AUTO) == ContactBackend.JAX_HASH
    assert resolve_contact_backend("auto") == ContactBackend.JAX_HASH
    assert normalize_contact_backend("jaxhash") == ContactBackend.JAX_HASH
    assert normalize_contact_backend("jax_hash") == ContactBackend.JAX_HASH
    assert resolve_contact_backend("jaxhash") == ContactBackend.JAX_HASH


def test_jaxhash_contact_batch_basic():
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)
    dummy_pair = np.array([0, 1], dtype=np.int32)

    capacity = 500
    cells, active, count, exhausted = jaxhash_contact_batch(
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


def test_jaxhash_contact_batch_capacity_exhausted():
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)
    dummy_pair = np.array([0, 1], dtype=np.int32)

    capacity = 1
    cells, active, count, exhausted = jaxhash_contact_batch(
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
        raise_if_jaxhash_contact_capacity_exhausted(capacity, count, exhausted)


def test_jaxhash_contact_matches_scipy():
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

    capacity = 15000
    cells, active, count, exhausted = jaxhash_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.0,
        rigid_contact_max=capacity,
    )

    assert not bool(exhausted)
    jax_active_cells = np.asarray(cells[active])

    def sort_contact(mat):
        mat = np.array(mat)
        mat = np.sort(mat, axis=1)
        idx = np.lexsort((mat[:, 1], mat[:, 0]))
        return mat[idx]

    s_sorted = sort_contact(scipy_cells)
    j_sorted = sort_contact(jax_active_cells)

    assert len(s_sorted) == len(j_sorted)
    np.testing.assert_array_equal(s_sorted, j_sorted)


def test_jaxhash_runs_on_cpu():
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 5, 50)])
        p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 5, 50)])
        points = np.vstack([p_fiber0, p_fiber1])
        fiber_ids = np.repeat(np.array([0, 1]), 50)
        diameters = np.full(100, 0.2)

        capacity = 500
        cells, active, count, exhausted = jaxhash_contact_batch(
            points=points,
            point_fiber_ids=fiber_ids,
            adjacency_block=2,
            point_diameters=diameters,
            search2radius_ratio=1.5,
            rigid_contact_max=capacity,
        )

        assert cells.devices() == {cpu_device}
        assert int(count) > 0
        assert not bool(exhausted)


def test_jaxhash_contact_batch_optional_rigid_contact_max():
    p_fiber0 = np.array([[0.0, 0.0, z] for z in np.linspace(0, 10, 100)])
    p_fiber1 = np.array([[0.1, 0.0, z] for z in np.linspace(0, 10, 100)])
    points = np.vstack([p_fiber0, p_fiber1])
    fiber_ids = np.repeat(np.array([0, 1]), 100)
    diameters = np.full(200, 0.2)

    # Call with rigid_contact_max=None (default)
    cells_auto, active_auto, count_auto, exhausted_auto = jaxhash_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.5,
        rigid_contact_max=None,
    )

    # Call with explicit rigid_contact_max
    cells_exp, active_exp, count_exp, exhausted_exp = jaxhash_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.5,
        rigid_contact_max=500,
    )

    assert int(count_auto) == int(count_exp)
    assert not bool(exhausted_auto)
    assert cells_auto.shape[0] >= int(np.ceil(int(count_auto) * 1.2))

    auto_active_pairs = np.asarray(cells_auto[active_auto])
    exp_active_pairs = np.asarray(cells_exp[active_exp])
    np.testing.assert_array_equal(auto_active_pairs, exp_active_pairs)


def test_cuda_hardware_info_query():
    info = get_cuda_hardware_info()
    assert "sm_count" in info
    assert "max_threads_per_sm" in info
    assert "hardware_concurrency" in info
    assert "cuda_cores" in info
    assert info["sm_count"] > 0
    assert info["hardware_concurrency"] > 0


def test_compute_dynamic_batch_params():
    N = 500_000
    C_max = 64
    n_batch, num_batches, hw_info = compute_dynamic_batch_params(N, C_max)

    assert n_batch > 0
    assert num_batches >= 1
    assert n_batch * num_batches >= N
    assert n_batch % hw_info["hardware_concurrency"] == 0 or n_batch == N


def test_jaxhash_explicit_node_batching_matches_unbatched():
    np.random.seed(123)
    N = 2000
    points = np.random.uniform(0, 5, size=(N, 3))
    fiber_ids = np.repeat(np.arange(20), 100)
    diameters = np.full(N, 0.4)
    dummy_pair = np.array([0, 1], dtype=np.int32)
    capacity = 10000

    # Run with default dynamic batching
    cells_auto, active_auto, count_auto, _ = jaxhash_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.1,
        rigid_contact_max=capacity,
        dummy_pair=dummy_pair,
        n_batch=None,
    )

    # Run with small forced batch size (e.g., 250 nodes per batch -> 8 batches)
    cells_batched, active_batched, count_batched, _ = jaxhash_contact_batch(
        points=points,
        point_fiber_ids=fiber_ids,
        adjacency_block=2,
        point_diameters=diameters,
        search2radius_ratio=1.1,
        rigid_contact_max=capacity,
        dummy_pair=dummy_pair,
        n_batch=250,
    )

    assert int(count_auto) == int(count_batched)
    pairs_auto = np.asarray(cells_auto[active_auto])
    pairs_batched = np.asarray(cells_batched[active_batched])
    np.testing.assert_array_equal(pairs_auto, pairs_batched)


