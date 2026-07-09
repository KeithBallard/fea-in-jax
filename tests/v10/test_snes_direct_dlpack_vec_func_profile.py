"""Profile buffer_callback vs direct-DLPack Vec wrappers with enough CUDA work.

Run normally:

    /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_snes_direct_dlpack_vec_func_profile.py 1000000 20

Run with Nsight Systems:

    nsys profile \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --sample=none \
      --cpuctxsw=none \
      --backtrace=none \
      --force-overwrite=true \
      -o petsc_jax_direct_vec_profile_heavy \
      /home/alberto/venvs/mpi-gpu/bin/python tests/v10/test_snes_direct_dlpack_vec_func_profile.py 1000000 20
"""

from __future__ import annotations

from contextlib import contextmanager
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

try:
    from cupyx.profiler import time_range as _cupy_time_range
except Exception:
    _cupy_time_range = None


def profiler_start():
    """Start a CUDA-profiler capture range if CuPy can reach cudart."""
    try:
        import cupy as cp

        cp.cuda.runtime.profilerStart()
    except Exception as exc:
        print("WARNING: cudaProfilerStart failed:", repr(exc))


def profiler_stop():
    """Stop a CUDA-profiler capture range if CuPy can reach cudart."""
    try:
        import cupy as cp

        cp.cuda.runtime.profilerStop()
    except Exception as exc:
        print("WARNING: cudaProfilerStop failed:", repr(exc))


def explicit_cupy_cuda_marker(n):
    """Force visible CuPy CUDA work inside the profiler range."""
    import cupy as cp

    with profile_range("explicit_cupy_cuda_marker"):
        x = cp.arange(min(n, 1_000_000), dtype=cp.float64)
        y = 2.0 * x + 1.0
        float(cp.sum(y).get())


@contextmanager
def profile_range(name):
    if _cupy_time_range is None:
        with jax.profiler.TraceAnnotation(name):
            yield
    else:
        with _cupy_time_range(name):
            yield


@jax.jit
def heavy_residual(x, args):
    scale, shift = args
    left = jnp.roll(x, 1)
    right = jnp.roll(x, -1)
    return scale * x + 0.01 * x * x + 0.1 * left - 0.07 * right + shift


def run_mode(name, *, use_buffer_callback, x0, args, repeats):
    X = jaxArrayToPETScVec(x0)
    F = X.duplicate()
    callback_func = convertJAXVecFuncToPETScVecFunc(
        heavy_residual,
        args,
        use_buffer_callback=use_buffer_callback,
    )

    elapsed_times = []
    checksum = None
    try:
        for i in range(repeats):
            with profile_range(f"{name}_repeat_{i + 1}"):
                t0 = time.perf_counter()
                callback_func(None, X, F, args)
                y = petscVecToJAX(F)
                checksum = jnp.sum(y).block_until_ready()
                elapsed = time.perf_counter() - t0
            elapsed_times.append(elapsed)
        print(f"\n=== {name} ===")
        print("checksum:", checksum)
        print("times:", elapsed_times)
        print("mean:", sum(elapsed_times) / len(elapsed_times))
        print("min:", min(elapsed_times))
        print("max:", max(elapsed_times))
        return elapsed_times
    finally:
        X.destroy()
        F.destroy()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    script_path = Path(__file__).as_posix()
    analyzer_path = (Path(__file__).resolve().parent / "analyze_nsys_memcopies.py").as_posix()

    x0 = jnp.linspace(0.1, 1.0, n, dtype=jnp.float64)
    args = jnp.array([2.0, 0.5], dtype=jnp.float64)

    print("Profiling heavy Vec wrapper paths.")
    print("vector size:", n)
    print("repeats:", repeats)

    print("\nWarmup JAX residual.")
    heavy_residual(x0, args).block_until_ready()

    print("\nWarmup wrappers.")
    run_mode(
        "warmup_buffer_callback",
        use_buffer_callback=True,
        x0=x0,
        args=args,
        repeats=1,
    )
    run_mode(
        "warmup_direct_dlpack",
        use_buffer_callback=False,
        x0=x0,
        args=args,
        repeats=1,
    )

    print("\nStarting cudaProfilerApi capture range.")
    profiler_start()
    try:
        explicit_cupy_cuda_marker(n)

        with profile_range("buffer_callback_vec_all_repeats"):
            buffer_times = run_mode(
                "buffer_callback_vec",
                use_buffer_callback=True,
                x0=x0,
                args=args,
                repeats=repeats,
            )

        with profile_range("direct_dlpack_vec_all_repeats"):
            direct_times = run_mode(
                "direct_dlpack_vec",
                use_buffer_callback=False,
                x0=x0,
                args=args,
                repeats=repeats,
            )
    finally:
        profiler_stop()
        print("\nStopped cudaProfilerApi capture range.")

    print("\n=== HEAVY DIRECT VEC WRAPPER SUMMARY ===")
    print("vector size:", n)
    print("repeats:", repeats)
    print("buffer mean:", sum(buffer_times) / len(buffer_times))
    print("direct mean:", sum(direct_times) / len(direct_times))
    print("buffer min:", min(buffer_times))
    print("direct min:", min(direct_times))
    print("Return this summary plus:")
    print("  nsys stats --force-export=true --report cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum petsc_jax_direct_vec_profile_heavy_api.nsys-rep")
    print("  nsys export --type sqlite --force true -o petsc_jax_direct_vec_profile_heavy_api.sqlite petsc_jax_direct_vec_profile_heavy_api.nsys-rep")
    print(f"  /home/alberto/venvs/mpi-gpu/bin/python {analyzer_path} petsc_jax_direct_vec_profile_heavy_api.sqlite")
    print("Profile command for this exact script location:")
    print("  nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none --backtrace=none --force-overwrite=true -o petsc_jax_direct_vec_profile_heavy_api \\")
    print(f"    /home/alberto/venvs/mpi-gpu/bin/python {script_path} {n} {repeats}")


if __name__ == "__main__":
    main()
