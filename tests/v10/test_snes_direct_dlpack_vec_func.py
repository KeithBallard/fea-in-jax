"""Compare buffer_callback and direct-DLPack JAX Vec-function wrappers.

This is for PETSc-as-outer-driver callbacks. PETSc provides X and F, JAX
computes values from X, and the wrapper mutates F.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_direct_dlpack_vec_func.py

Optional Nsight profile:

    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --backtrace=none \
      --force-overwrite=true \
      -o petsc_jax_direct_vec_profile \
      /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_direct_dlpack_vec_func.py
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

from v10.NonlinearSNES import (
    convertJAXVecFuncToPETScVecFunc,
    jaxArrayToPETScVec,
    petscVecToJAX,
)


jax.config.update("jax_enable_x64", True)


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


def check_callback_mode(name, *, use_buffer_callback):
    args = jnp.array([4.0, 1.0, 27.0], dtype=jnp.float64)
    x0 = jnp.array([5.0, 5.0, 5.0, 0.25, 2.0, 3.0], dtype=jnp.float64)
    expected = example_residual(x0, args)

    X = jaxArrayToPETScVec(x0)
    F = X.duplicate()
    expected_vec = jaxArrayToPETScVec(expected)
    diff = None
    try:
        callback_func = convertJAXVecFuncToPETScVecFunc(
            example_residual,
            args,
            use_buffer_callback=use_buffer_callback,
        )

        t0 = time.perf_counter()
        callback_func(None, X, F, args)
        elapsed = time.perf_counter() - t0

        output = petscVecToJAX(F)
        print(f"\n=== {name} ===")
        print("elapsed:", elapsed)
        print("output:", output)
        print("expected:", expected)

        diff = F.duplicate()
        F.copy(diff)
        diff.axpy(-1.0, expected_vec)
        error_norm = diff.norm()
        print("PETSc-space error norm:", error_norm)
        assert error_norm < 1e-12
        assert jnp.allclose(output, expected)
        return elapsed
    finally:
        if diff is not None:
            diff.destroy()
        X.destroy()
        F.destroy()
        expected_vec.destroy()


def main():
    buffer_elapsed = check_callback_mode(
        "buffer_callback Vec wrapper",
        use_buffer_callback=True,
    )
    direct_elapsed = check_callback_mode(
        "direct-DLPack Vec wrapper",
        use_buffer_callback=False,
    )

    print("\n=== DIRECT VEC WRAPPER SUMMARY ===")
    print("buffer_callback elapsed:", buffer_elapsed)
    print("direct-DLPack elapsed:", direct_elapsed)


if __name__ == "__main__":
    main()
