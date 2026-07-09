"""Smoke test for dense JAX Jacobian -> COOData conversion.

Run from the PETScJVP root with the PETSc/JAX environment, for example:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_jacfwd_to_coo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.NonlinearSNES import convertJaxMatToCOOData


jax.config.update("jax_enable_x64", True)


def example_residual(x):
    return jnp.array(
        [
            x[0] ** 2 + 3.0 * x[1] - x[2],
            x[0] * x[1] + jnp.sin(x[2]),
            x[2] ** 3 - 2.0 * x[0],
        ],
        dtype=x.dtype,
    )


def main():
    x0 = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float64)
    jac = jax.jacfwd(example_residual)(x0)
    coo_data = convertJaxMatToCOOData(jac)

    expected_jac = jnp.array(
        [
            [4.0, 3.0, -1.0],
            [-1.0, 2.0, jnp.cos(0.5)],
            [-2.0, 0.0, 0.75],
        ],
        dtype=jnp.float64,
    )

    print("Testing jax.jacfwd residual Jacobian -> COOData.")
    print("x0:", x0)
    print("dense jacobian:")
    print(jac)
    print("COO shape:", coo_data.shape)
    print("COO rows:", coo_data.rows)
    print("COO cols:", coo_data.cols)
    print("COO vals:", coo_data.vals)

    assert coo_data.shape.tolist() == [3, 3]
    assert coo_data.rows.dtype == jnp.int32
    assert coo_data.cols.dtype == jnp.int32
    assert coo_data.vals.dtype == jac.dtype
    assert coo_data.vals.shape == (9,)

    reconstructed = jnp.zeros(tuple(coo_data.shape.tolist()), dtype=coo_data.vals.dtype)
    reconstructed = reconstructed.at[coo_data.rows, coo_data.cols].set(coo_data.vals)

    print("reconstructed dense jacobian:")
    print(reconstructed)
    print("expected dense jacobian:")
    print(expected_jac)

    assert jnp.allclose(jac, expected_jac)
    assert jnp.allclose(reconstructed, jac)


if __name__ == "__main__":
    main()
