"""Smoke test for the direct-only PETSc Vec function wrapper.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_snes_direct_only_vec_func.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.NonlinearSNES.direct_vec_function_converters import (
    convertJAXVecFuncToPETScVecFuncDirect,
    jaxArrayToPETScVec,
    petscVecToJAX,
)


jax.config.update("jax_enable_x64", True)


@jax.jit
def example_residual(x, args):
    return jnp.array(
        [
            x[0] ** 2 - args[0],
            x[1] ** 2 - args[1],
            x[2] ** 3 - args[2],
            x[0] + x[1] + x[2],
            jnp.sin(x[3]),
            x[4] * x[5],
        ],
        dtype=x.dtype,
    )


def main():
    args = jnp.array([4.0, 1.0, 27.0], dtype=jnp.float64)
    x0 = jnp.array([5.0, 5.0, 5.0, 0.25, 2.0, 3.0], dtype=jnp.float64)
    expected = example_residual(x0, args)

    X = jaxArrayToPETScVec(x0)
    F = X.duplicate()
    expected_vec = jaxArrayToPETScVec(expected)
    diff = None
    try:
        petsc_func = convertJAXVecFuncToPETScVecFuncDirect(example_residual, args)

        t0 = time.perf_counter()
        petsc_func(None, X, F, args)
        elapsed = time.perf_counter() - t0

        output = petscVecToJAX(F)
        diff = F.duplicate()
        F.copy(diff)
        diff.axpy(-1.0, expected_vec)
        error_norm = diff.norm()

        print("Testing direct-only JAX Vec function -> PETSc Vec callback.")
        print("elapsed:", elapsed)
        print("output:", output)
        print("expected:", expected)
        print("PETSc-space error norm:", error_norm)

        assert error_norm < 1e-12
        assert jnp.allclose(output, expected)
    finally:
        if diff is not None:
            diff.destroy()
        X.destroy()
        F.destroy()
        expected_vec.destroy()


if __name__ == "__main__":
    main()
