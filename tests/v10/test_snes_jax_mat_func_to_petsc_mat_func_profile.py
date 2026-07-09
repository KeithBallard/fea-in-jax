"""Profile JAX matrix function -> PETSc Mat callback conversion.

Run normally:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_jax_mat_func_to_petsc_mat_func_profile.py 256 3

Run with Nsight Systems:

    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --backtrace=none \
      --force-overwrite=true \
      -o petsc_jax_mat_func_profile \
      /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_jax_mat_func_to_petsc_mat_func_profile.py 256 3
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from petsc4py import PETSc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.NonlinearSNES import (
    convertJAXMatFuncToPETScMatFunc,
    jaxArrayToPETScVec,
)


jax.config.update("jax_enable_x64", True)

try:
    from cupyx.profiler import time_range as _cupy_time_range
except Exception:
    _cupy_time_range = None


@contextmanager
def profile_range(name):
    if _cupy_time_range is None:
        with jax.profiler.TraceAnnotation(name):
            yield
    else:
        with _cupy_time_range(name):
            yield


def nonlinear_residual(x):
    """Dense-ish nonlinear residual whose jacfwd result is easy to matvec."""
    n = x.shape[0]
    idx = jnp.arange(n, dtype=x.dtype)
    diag = 2.0 + 0.01 * idx
    left = jnp.roll(x, 1)
    right = jnp.roll(x, -1)
    global_coupling = 0.001 * jnp.sum(jnp.sin(x))
    return diag * x + 0.05 * x**2 + 0.1 * left - 0.07 * right + global_coupling


def _destroy_all(*objects):
    for obj in objects:
        if obj is not None:
            obj.destroy()


def run_one(jac_func, x0, test_vec, *, warmup=False):
    with profile_range("profile_jacfwd_dense_matrix"):
        dense_jac = jac_func(x0)
    with profile_range("profile_dense_jacobian_matvec_expected"):
        expected_action = dense_jac @ test_vec

    with profile_range("profile_create_petsc_objects"):
        X = jaxArrayToPETScVec(x0)
        V = jaxArrayToPETScVec(test_vec)
        expected_vec = jaxArrayToPETScVec(expected_action)
        Y = expected_vec.duplicate()
        mat = PETSc.Mat().create(PETSc.COMM_WORLD)

    diff = None
    try:
        callback_func = convertJAXMatFuncToPETScMatFunc(jac_func)

        t0 = time.perf_counter()
        with profile_range("profile_jax_mat_func_to_petsc_mat_callback"):
            callback_func(None, X, mat, mat, None)
        with profile_range("profile_petsc_mat_assemble"):
            mat.assemble()
        with profile_range("profile_petsc_mat_mult"):
            mat.mult(V, Y)
        PETSc.COMM_WORLD.barrier()
        elapsed = time.perf_counter() - t0

        with profile_range("profile_petsc_action_error"):
            diff = Y.duplicate()
            Y.copy(diff)
            diff.axpy(-1.0, expected_vec)
            error_norm = diff.norm()

        if not warmup:
            print("action error norm:", error_norm)
        assert error_norm < 1e-9
        return elapsed
    finally:
        _destroy_all(diff, X, V, expected_vec, Y, mat)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    x0 = jnp.linspace(0.1, 1.0, n, dtype=jnp.float64)
    test_vec = jnp.cos(jnp.linspace(0.0, 3.0, n, dtype=jnp.float64))
    jac_func = jax.jacfwd(nonlinear_residual)

    print("Profiling JAX matrix function -> PETSc Mat callback.")
    print("matrix size:", n, "x", n)
    print("dense entries:", n * n)
    print("timing repeats:", repeats)

    print("\nWarmup.")
    warmup_elapsed = run_one(jac_func, x0, test_vec, warmup=True)
    print("warmup elapsed:", warmup_elapsed)

    elapsed_times = []
    print("\nProfiled repeats.")
    for i in range(repeats):
        with profile_range(f"profile_repeat_{i + 1}"):
            elapsed = run_one(jac_func, x0, test_vec)
        elapsed_times.append(elapsed)
        print(f"repeat {i + 1} elapsed: {elapsed:.6f} s")

    print("\n=== PROFILE TEST SUMMARY ===")
    print("matrix size:", n, "x", n)
    print("dense entries:", n * n)
    print("repeats:", repeats)
    print("elapsed times:", elapsed_times)
    print("min elapsed:", min(elapsed_times))
    print("mean elapsed:", sum(elapsed_times) / len(elapsed_times))
    print("max elapsed:", max(elapsed_times))
    print("Return this summary plus:")
    print("  nsys stats --report cuda_api_sum petsc_jax_mat_func_profile.nsys-rep")
    print("  nsys stats --report cuda_gpu_mem_time_sum petsc_jax_mat_func_profile.nsys-rep")
    print("  nsys stats --report cuda_gpu_mem_size_sum petsc_jax_mat_func_profile.nsys-rep")
    print("  optionally: a screenshot of the NVTX timeline around snes_* ranges")


if __name__ == "__main__":
    main()
