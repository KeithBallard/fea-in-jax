import numpy as np
import pytest
from jax import numpy as jnp

from fe_jax.contact_jitless import (
    ContactPointPair,
    canonicalize_contact_point_pair,
    contact_batch,
    distinct_fiber_node2node,
    duplicate_filtering,
    self_fiber_node2node,
)


def test_canonicalize_contact_point_pair_keeps_order_when_already_canonical():
    pair = canonicalize_contact_point_pair(
        fiber_i=1,
        fiber_j=3,
        node_i=2,
        node_j=4,
        x_i=jnp.array([1.0, 0.0, 0.0]),
        x_j=jnp.array([3.0, 0.0, 0.0]),
        distance=2.0,
    )

    assert pair.fiber_i == 1
    assert pair.fiber_j == 3
    assert pair.node_i == 2
    assert pair.node_j == 4
    np.testing.assert_allclose(pair.x_i, jnp.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(pair.x_j, jnp.array([3.0, 0.0, 0.0]))
    assert pair.distance == 2.0


def test_canonicalize_contact_point_pair_swaps_order_when_needed():
    pair = canonicalize_contact_point_pair(
        fiber_i=4,
        fiber_j=2,
        node_i=7,
        node_j=3,
        x_i=jnp.array([4.0, 0.0, 0.0]),
        x_j=jnp.array([2.0, 0.0, 0.0]),
        distance=2.0,
    )

    assert pair.fiber_i == 2
    assert pair.fiber_j == 4
    assert pair.node_i == 3
    assert pair.node_j == 7
    np.testing.assert_allclose(pair.x_i, jnp.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(pair.x_j, jnp.array([4.0, 0.0, 0.0]))
    assert pair.distance == 2.0


def test_duplicate_filtering_removes_duplicates_and_keeps_first_occurrence():
    pair_a = canonicalize_contact_point_pair(
        fiber_i=0,
        fiber_j=1,
        node_i=0,
        node_j=2,
        x_i=jnp.array([0.0, 0.0, 0.0]),
        x_j=jnp.array([1.0, 0.0, 0.0]),
        distance=1.0,
    )
    pair_a_duplicate = ContactPointPair(
        fiber_i=0,
        fiber_j=1,
        node_i=0,
        node_j=2,
        x_i=jnp.array([0.0, 1.0, 0.0]),
        x_j=jnp.array([1.0, 1.0, 0.0]),
        distance=9.0,
    )
    pair_b = canonicalize_contact_point_pair(
        fiber_i=0,
        fiber_j=2,
        node_i=1,
        node_j=3,
        x_i=jnp.array([0.0, 0.0, 1.0]),
        x_j=jnp.array([1.0, 0.0, 1.0]),
        distance=1.0,
    )

    filtered = duplicate_filtering([pair_a, pair_a_duplicate, pair_b])

    assert len(filtered) == 2
    assert filtered[0] is pair_a
    assert filtered[1] is pair_b


def test_duplicate_filtering_rejects_non_canonical_contacts():
    bad_pair = ContactPointPair(
        fiber_i=3,
        fiber_j=1,
        node_i=5,
        node_j=2,
        x_i=jnp.array([0.0, 0.0, 0.0]),
        x_j=jnp.array([1.0, 0.0, 0.0]),
        distance=1.0,
    )

    with pytest.raises(AssertionError):
        duplicate_filtering([bad_pair])


def test_distinct_fiber_node2node_finds_expected_pair():
    fiber_a = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    fiber_b = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    contacts = distinct_fiber_node2node(
        fiber_x=0,
        x_nd=fiber_a,
        fiber_y=1,
        y_nd=fiber_b,
        radius=0.25,
    )

    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.fiber_i == 0
    assert contact.fiber_j == 1
    assert contact.node_i == 0
    assert contact.node_j == 0
    np.testing.assert_allclose(contact.x_i, fiber_a[0])
    np.testing.assert_allclose(contact.x_j, fiber_b[0])
    assert float(contact.distance) == pytest.approx(0.1)


def test_self_fiber_node2node_excludes_neighboring_nodes_and_keeps_far_contact():
    fiber = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )

    contacts = self_fiber_node2node(
        fiber_x=5,
        x_nd=fiber,
        radius=0.5,
    )

    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.fiber_i == 5
    assert contact.fiber_j == 5
    assert contact.node_i == 0
    assert contact.node_j == 3
    np.testing.assert_allclose(contact.x_i, fiber[0])
    np.testing.assert_allclose(contact.x_j, fiber[3])
    assert float(contact.distance) == pytest.approx(0.1)


def test_contact_batch_uses_injected_detectors_and_orders_results():
    fibers = [
        jnp.array([[0.0, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    ]

    def distinct_stub(*, fiber_x, x_nd, fiber_y, y_nd, radius):
        return [
            canonicalize_contact_point_pair(
                fiber_i=fiber_x,
                fiber_j=fiber_y,
                node_i=0,
                node_j=0,
                x_i=x_nd[0],
                x_j=y_nd[0],
                distance=radius,
            )
        ]

    def self_stub(*, fiber_x, x_nd, radius):
        return [
            canonicalize_contact_point_pair(
                fiber_i=fiber_x,
                fiber_j=fiber_x,
                node_i=0,
                node_j=0,
                x_i=x_nd[0],
                x_j=x_nd[0],
                distance=radius,
            )
        ]

    contacts = contact_batch(
        fibers=fibers,
        radius=0.5,
        distinct_fiber_fn=distinct_stub,
        self_fiber_fn=self_stub,
    )

    assert len(contacts) == 3
    assert contacts[0].fiber_i == 0
    assert contacts[0].fiber_j == 1
    assert contacts[1].fiber_i == 0
    assert contacts[1].fiber_j == 0
    assert contacts[2].fiber_i == 1
    assert contacts[2].fiber_j == 1


def test_contact_batch_rejects_non_positive_radius():
    fibers = [jnp.array([[0.0, 0.0, 0.0]])]

    with pytest.raises(ValueError, match="radius must be positive"):
        contact_batch(fibers=fibers, radius=0.0)


def test_contact_batch_rejects_bad_fiber_shape():
    fibers = [jnp.array([[0.0, 0.0], [1.0, 1.0]])]

    with pytest.raises(ValueError, match="Fiber 0 must have shape"):
        contact_batch(fibers=fibers, radius=0.1)
