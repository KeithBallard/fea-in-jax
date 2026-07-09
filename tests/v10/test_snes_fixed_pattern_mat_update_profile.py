"""Profile repeated PETSc Mat updates with and without COO preallocation.

This test checks whether skipping `setSizes`, `setType`, and
`setPreallocationCOO` removes the repeated HtoD/DtoH traffic observed in the
dense JAX matrix-function path.

Run normally:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_fixed_pattern_mat_update_profile.py 256 3

Run with Nsight Systems:

    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --backtrace=none \
      --force-overwrite=true \
      -o petsc_jax_fixed_pattern_profile \
      /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_fixed_pattern_mat_update_profile.py 256 3
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


def _check_mat_action(mat, x, test_vec, jac_func, *, range_prefix):
    with profile_range(f"{range_prefix}_expected_action"):
        expected_action = jac_func(x) @ test_vec

    V = jaxArrayToPETScVec(test_vec)
    expected_vec = jaxArrayToPETScVec(expected_action)
    Y = expected_vec.duplicate()
    diff = None
    try:
        with profile_range(f"{range_prefix}_petsc_mat_mult"):
            mat.mult(V, Y)
        with profile_range(f"{range_prefix}_action_error"):
            diff = Y.duplicate()
            Y.copy(diff)
            diff.axpy(-1.0, expected_vec)
            error_norm = diff.norm()
        assert error_norm < 1e-9
        return error_norm
    finally:
        _destroy_all(diff, V, expected_vec, Y)


def run_rebuild_each_time(jac_func, xs, test_vec):
    callback_func = convertJAXMatFuncToPETScMatFunc(jac_func, set_preallocation=True)
    elapsed_times = []

    for step, x in enumerate(xs):
        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        X = jaxArrayToPETScVec(x)
        try:
            with profile_range(f"rebuild_step_{step + 1}"):
                t0 = time.perf_counter()
                with profile_range("rebuild_callback_with_preallocation"):
                    callback_func(None, X, mat, mat, None)
                error_norm = _check_mat_action(
                    mat,
                    x,
                    test_vec,
                    jac_func,
                    range_prefix="rebuild",
                )
                PETSc.COMM_WORLD.barrier()
                elapsed = time.perf_counter() - t0
            elapsed_times.append(elapsed)
            print(f"rebuild step {step + 1}: {elapsed:.6f} s, error {error_norm}")
        finally:
            _destroy_all(X, mat)

    return elapsed_times


def run_fixed_pattern_update(jac_func, xs, test_vec):
    preallocate_callback = convertJAXMatFuncToPETScMatFunc(jac_func, set_preallocation=True)
    update_callback = convertJAXMatFuncToPETScMatFunc(jac_func, set_preallocation=False)

    mat = PETSc.Mat().create(PETSc.COMM_WORLD)
    elapsed_times = []
    try:
        for step, x in enumerate(xs):
            X = jaxArrayToPETScVec(x)
            try:
                with profile_range(f"fixed_pattern_step_{step + 1}"):
                    t0 = time.perf_counter()
                    if step == 0:
                        with profile_range("fixed_pattern_initial_preallocation"):
                            preallocate_callback(None, X, mat, mat, None)
                    else:
                        with profile_range("fixed_pattern_values_only_update"):
                            update_callback(None, X, mat, mat, None)

                    error_norm = _check_mat_action(
                        mat,
                        x,
                        test_vec,
                        jac_func,
                        range_prefix="fixed_pattern",
                    )
                    PETSc.COMM_WORLD.barrier()
                    elapsed = time.perf_counter() - t0
                elapsed_times.append(elapsed)
                print(f"fixed-pattern step {step + 1}: {elapsed:.6f} s, error {error_norm}")
            finally:
                X.destroy()
    finally:
        mat.destroy()

    return elapsed_times


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    x0 = jnp.linspace(0.1, 1.0, n, dtype=jnp.float64)
    xs = [x0 + (0.01 * i) for i in range(repeats)]
    test_vec = jnp.cos(jnp.linspace(0.0, 3.0, n, dtype=jnp.float64))
    jac_func = jax.jacfwd(nonlinear_residual)

    print("Profiling rebuild-vs-fixed-pattern PETSc Mat updates.")
    print("matrix size:", n, "x", n)
    print("dense entries:", n * n)
    print("steps:", repeats)

    print("\nWarmup.")
    warmup_xs = [x0]
    run_rebuild_each_time(jac_func, warmup_xs, test_vec)
    run_fixed_pattern_update(jac_func, warmup_xs, test_vec)

    print("\nRebuild Mat each step.")
    with profile_range("rebuild_all_steps"):
        rebuild_times = run_rebuild_each_time(jac_func, xs, test_vec)

    print("\nFixed-pattern Mat update.")
    with profile_range("fixed_pattern_all_steps"):
        fixed_times = run_fixed_pattern_update(jac_func, xs, test_vec)

    print("\n=== FIXED PATTERN PROFILE SUMMARY ===")
    print("matrix size:", n, "x", n)
    print("steps:", repeats)
    print("rebuild times:", rebuild_times)
    print("fixed-pattern times:", fixed_times)
    print("rebuild mean:", sum(rebuild_times) / len(rebuild_times))
    print("fixed-pattern mean:", sum(fixed_times) / len(fixed_times))
    if len(fixed_times) > 1:
        update_only = fixed_times[1:]
        print("fixed-pattern update-only times:", update_only)
        print("fixed-pattern update-only mean:", sum(update_only) / len(update_only))
    print("Return this summary plus:")
    print("  nsys stats --report cuda_gpu_mem_time_sum,nvtx_sum petsc_jax_fixed_pattern_profile.nsys-rep")
    print("  nsys export --type sqlite --force true -o petsc_jax_fixed_pattern_profile.sqlite petsc_jax_fixed_pattern_profile.nsys-rep")
    print("  /home/alberto/venvs/mpi-gpu/bin/python v10/analyze_nsys_memcopies.py petsc_jax_fixed_pattern_profile.sqlite")


if __name__ == "__main__":
    main()
