import numpy as np
import pytest
from jax import numpy as jnp

from fe_jax.contact import (
    contact_batch,
    count_initial_contacts,
    distinct_fiber_node2node,
    # merge_contact_cells,
    self_fiber_node2node,
)


def test_count_initial_contacts_counts_distinct_and_self_contacts():
    points = jnp.array(
        [
            [0.0, 0.0, 0.0],   # fiber 0, node 0
            [0.0, 0.1, 0.0],   # fiber 1, node 0
            [2.0, 0.0, 0.0],   # fiber 0, node 1
            [2.0, 0.1, 0.0],   # fiber 1, node 1
            [0.0, 0.0, 1.0],   # fiber 2, node 0
            [0.0, 0.0, 1.1],   # fiber 2, node 1
            [0.0, 0.0, 1.2],   # fiber 2, node 2
            [0.0, 0.0, 1.3],   # fiber 2, node 3
        ]
    )
    point_fiber_ids = jnp.array([0, 1, 0, 1, 2, 2, 2, 2])

    # distinct contacts: (0,1) and (2,3)
    # self contacts on fiber 2: (4,7)
    count = count_initial_contacts(
        points=points,
        point_fiber_ids=point_fiber_ids,
        radius=0.15,
        adjacency_block=2,
    )

    assert int(count) == 2


# def test_merge_contact_cells_packs_valid_rows_before_sentinels():
#     distinct_contacts = jnp.array(
#         [
#             [0, 1],
#             [0, 0],
#             [0, 0],
#         ],
#         dtype=jnp.uint64,
#     )
#     self_contacts = jnp.array(
#         [
#             [2, 3],
#             [0, 0],
#             [0, 0],
#         ],
#         dtype=jnp.uint64,
#     )

#     merged, n_contact, overflowed = merge_contact_cells(
#         distinct_contacts=distinct_contacts,
#         n_distinct=1,
#         self_contacts=self_contacts,
#         n_self=1,
#         capacity=3,
#     )

#     np.testing.assert_array_equal(np.asarray(merged), np.array([[0, 1], [2, 3], [0, 0]], dtype=np.uint64))
#     assert int(n_contact) == 2
#     assert bool(overflowed) is False


# def test_merge_contact_cells_flags_overflow():
#     distinct_contacts = jnp.array(
#         [
#             [0, 1],
#             [2, 3],
#         ],
#         dtype=jnp.uint64,
#     )
#     self_contacts = jnp.array(
#         [
#             [4, 5],
#             [6, 7],
#         ],
#         dtype=jnp.uint64,
#     )

#     merged, n_contact, overflowed = merge_contact_cells(
#         distinct_contacts=distinct_contacts,
#         n_distinct=2,
#         self_contacts=self_contacts,
#         n_self=2,
#         capacity=3,
#     )

#     np.testing.assert_array_equal(np.asarray(merged), np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint64))
#     assert int(n_contact) == 3
#     assert bool(overflowed) is True


def test_distinct_fiber_node2node_finds_expected_pair():
    points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    point_fiber_ids = jnp.array([0, 0, 1, 1])

    contacts = distinct_fiber_node2node(
        points=points,
        point_fiber_ids=point_fiber_ids,
        radius=0.25,
    )

    np.testing.assert_array_equal(np.asarray(contacts), np.array([[0, 2]], dtype=np.int32))


def test_distinct_fiber_node2node_accepts_1d_points():
    points = jnp.array([[0.0], [1.0], [0.1], [10.0]])
    point_fiber_ids = jnp.array([0, 0, 1, 1])

    contacts = distinct_fiber_node2node(
        points=points,
        point_fiber_ids=point_fiber_ids,
        radius=0.25,
    )

    np.testing.assert_array_equal(np.asarray(contacts), np.array([[0, 2]], dtype=np.int32))


def test_self_fiber_node2node_excludes_neighbors_and_keeps_far_pair():
    points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    point_fiber_ids = jnp.array([5, 5, 5, 5])

    contacts = self_fiber_node2node(
        points=points,
        point_fiber_ids=point_fiber_ids,
        radius=0.5,
        adjacency_block=1,
    )

    np.testing.assert_array_equal(np.asarray(contacts), np.array([[0, 3]], dtype=np.int32))


def test_self_fiber_node2node_accepts_2d_points():
    points = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 0.1],
        ]
    )
    point_fiber_ids = jnp.array([5, 5, 5, 5])

    contacts = self_fiber_node2node(
        points=points,
        point_fiber_ids=point_fiber_ids,
        radius=0.5,
        adjacency_block=1,
    )

    np.testing.assert_array_equal(np.asarray(contacts), np.array([[0, 3]], dtype=np.int32))


def test_contact_batch_concatenates_detector_outputs_and_forwards_arguments():
    points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    point_fiber_ids = jnp.array([0, 1])

    distinct_calls = []
    self_calls = []

    def distinct_stub(*, points, point_fiber_ids, radius):
        distinct_calls.append((np.asarray(points), np.asarray(point_fiber_ids), radius))
        return jnp.array([[0, 1]], dtype=jnp.int32)

    def self_stub(*, points, point_fiber_ids, radius, adjacency_block):
        self_calls.append((np.asarray(points), np.asarray(point_fiber_ids), radius, adjacency_block))
        return jnp.array([[1, 1]], dtype=jnp.int32)

    contacts = contact_batch(
        points=points,
        point_fiber_ids=point_fiber_ids,
        adjacency_block=3,
        radius=0.5,
        distinct_fiber_fn=distinct_stub,
        self_fiber_fn=self_stub,
    )

    np.testing.assert_array_equal(np.asarray(contacts), np.array([[0, 1], [1, 1]], dtype=np.int32))
    assert len(distinct_calls) == 1
    assert len(self_calls) == 1
    np.testing.assert_array_equal(distinct_calls[0][0], np.asarray(points))
    np.testing.assert_array_equal(distinct_calls[0][1], np.asarray(point_fiber_ids))
    assert distinct_calls[0][2] == 0.5
    np.testing.assert_array_equal(self_calls[0][0], np.asarray(points))
    np.testing.assert_array_equal(self_calls[0][1], np.asarray(point_fiber_ids))
    assert self_calls[0][2] == 0.5
    assert self_calls[0][3] == 3


def test_contact_batch_rejects_non_positive_radius():
    points = jnp.array([[0.0, 0.0, 0.0]])
    point_fiber_ids = jnp.array([0])

    with pytest.raises(ValueError, match="radius must be positive"):
        contact_batch(
            points=points,
            point_fiber_ids=point_fiber_ids,
            adjacency_block=1,
            radius=0.0,
        )


def test_contact_batch_rejects_bad_point_shape():
    points = jnp.array([[0.0, 0.0, 0.0, 1.0]])
    point_fiber_ids = jnp.array([0])

    with pytest.raises(ValueError, match="points must have shape"):
        contact_batch(
            points=points,
            point_fiber_ids=point_fiber_ids,
            adjacency_block=1,
            radius=0.5,
        )
