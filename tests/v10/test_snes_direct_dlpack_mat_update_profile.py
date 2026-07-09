"""Compare buffer_callback and direct-DLPack PETSc Mat value updates.

This test is for PETSc-as-outer-driver workflows. It checks whether, outside a
JAX-staged call, we can avoid the buffer_callback boundary by converting the
JAX output array to a CuPy view directly and passing that pointer to PETSc.

Run normally:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_direct_dlpack_mat_update_profile.py 256 3

Run with Nsight Systems:

    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --backtrace=none \
      --force-overwrite=true \
      -o petsc_jax_direct_dlpack_profile \
      /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_direct_dlpack_mat_update_profile.py 256 3
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


def _run_persistent_path(
    *,
    path_name,
    jac_func,
    xs,
    test_vec,
    use_buffer_callback,
):
    preallocate_callback = convertJAXMatFuncToPETScMatFunc(
        jac_func,
        set_preallocation=True,
        use_buffer_callback=use_buffer_callback,
    )
    update_callback = convertJAXMatFuncToPETScMatFunc(
        jac_func,
        set_preallocation=False,
        use_buffer_callback=use_buffer_callback,
    )

    mat = PETSc.Mat().create(PETSc.COMM_WORLD)
    elapsed_times = []
    try:
        for step, x in enumerate(xs):
            X = jaxArrayToPETScVec(x)
            try:
                with profile_range(f"{path_name}_step_{step + 1}"):
                    t0 = time.perf_counter()
                    if step == 0:
                        with profile_range(f"{path_name}_initial_preallocation"):
                            preallocate_callback(None, X, mat, mat, None)
                    else:
                        with profile_range(f"{path_name}_values_only_update"):
                            update_callback(None, X, mat, mat, None)

                    error_norm = _check_mat_action(
                        mat,
                        x,
                        test_vec,
                        jac_func,
                        range_prefix=path_name,
                    )
                    PETSc.COMM_WORLD.barrier()
                    elapsed = time.perf_counter() - t0
                elapsed_times.append(elapsed)
                print(f"{path_name} step {step + 1}: {elapsed:.6f} s, error {error_norm}")
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

    print("Profiling buffer_callback vs direct-DLPack PETSc Mat updates.")
    print("matrix size:", n, "x", n)
    print("dense entries:", n * n)
    print("steps:", repeats)

    print("\nWarmup.")
    _run_persistent_path(
        path_name="warmup_buffer_callback",
        jac_func=jac_func,
        xs=[x0],
        test_vec=test_vec,
        use_buffer_callback=True,
    )
    _run_persistent_path(
        path_name="warmup_direct_dlpack",
        jac_func=jac_func,
        xs=[x0],
        test_vec=test_vec,
        use_buffer_callback=False,
    )

    print("\nBuffer-callback persistent Mat updates.")
    with profile_range("buffer_callback_all_steps"):
        buffer_times = _run_persistent_path(
            path_name="buffer_callback",
            jac_func=jac_func,
            xs=xs,
            test_vec=test_vec,
            use_buffer_callback=True,
        )

    print("\nDirect-DLPack persistent Mat updates.")
    with profile_range("direct_dlpack_all_steps"):
        direct_times = _run_persistent_path(
            path_name="direct_dlpack",
            jac_func=jac_func,
            xs=xs,
            test_vec=test_vec,
            use_buffer_callback=False,
        )

    print("\n=== DIRECT DLPACK PROFILE SUMMARY ===")
    print("matrix size:", n, "x", n)
    print("steps:", repeats)
    print("buffer-callback times:", buffer_times)
    print("direct-DLPack times:", direct_times)
    print("buffer-callback mean:", sum(buffer_times) / len(buffer_times))
    print("direct-DLPack mean:", sum(direct_times) / len(direct_times))
    if len(buffer_times) > 1 and len(direct_times) > 1:
        buffer_update = buffer_times[1:]
        direct_update = direct_times[1:]
        print("buffer-callback update-only times:", buffer_update)
        print("direct-DLPack update-only times:", direct_update)
        print("buffer-callback update-only mean:", sum(buffer_update) / len(buffer_update))
        print("direct-DLPack update-only mean:", sum(direct_update) / len(direct_update))
    print("Return this summary plus:")
    print("  nsys stats --report cuda_gpu_mem_time_sum,nvtx_sum petsc_jax_direct_dlpack_profile.nsys-rep")
    print("  nsys export --type sqlite --force true -o petsc_jax_direct_dlpack_profile.sqlite petsc_jax_direct_dlpack_profile.nsys-rep")
    print("  /home/alberto/venvs/mpi-gpu/bin/python v10/analyze_nsys_memcopies.py petsc_jax_direct_dlpack_profile.sqlite")


if __name__ == "__main__":
    main()
