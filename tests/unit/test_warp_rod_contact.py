import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fe_jax.contact.warp_rod_contact import (
    NEWTON_WARP_AVAILABLE,
    build_newton_rods_contact,
    NewtonRodContactSearch,
    RodContactResult,
)
from fe_jax.helper import read_fabric


@pytest.mark.skipif(not NEWTON_WARP_AVAILABLE, reason="Newton/Warp not installed")
def test_warp_rod_contact_basic():
    fabric = read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab")
    capacity = 15000
    ctx = build_newton_rods_contact(fabric, rigid_contact_max=capacity)

    points_jax = jnp.asarray(fabric.points, dtype=jnp.float32)

    # 1. Test search
    res = NewtonRodContactSearch(ctx, points_jax)
    assert isinstance(res, RodContactResult)
    assert res.seg0.shape == (capacity,)
    assert res.seg1.shape == (capacity,)
    assert res.point0.shape == (capacity, 3)
    assert res.point1.shape == (capacity, 3)
    assert res.normal.shape == (capacity, 3)
    assert res.active.shape == (capacity,)
    assert int(res.count) > 0

    # 2. Test inside @jax.jit
    @jax.jit
    def jitted_search(pts):
        return NewtonRodContactSearch(ctx, pts)

    res_jit = jitted_search(points_jax)
    jax.block_until_ready(res_jit)
    assert int(res_jit.count) == int(res.count)
    assert int(jnp.sum(res_jit.active)) == int(res_jit.count)


@pytest.mark.skipif(not NEWTON_WARP_AVAILABLE, reason="Newton/Warp not installed")
def test_warp_rod_contact_displacement():
    fabric = read_fabric("experiments/initial_single_fiber/initial_single_fiber.fab")
    ctx = build_newton_rods_contact(fabric, rigid_contact_max=1000)

    # Separate points far away so no contacts occur
    points_far = jnp.asarray(fabric.points, dtype=jnp.float32)
    offsets = jnp.zeros_like(points_far)
    offsets = offsets.at[:, 2].set(jnp.arange(len(points_far)) * 10.0)
    points_far = points_far + offsets

    @jax.jit
    def jitted_search(pts):
        return NewtonRodContactSearch(ctx, pts)

    res = jitted_search(points_far)
    jax.block_until_ready(res)
    assert int(res.count) == 0
    assert int(jnp.sum(res.active)) == 0
