import numpy as np

import jax
from petsc4py import PETSc


def _comm_for_ksp(ksp):
    return getattr(ksp, "comm", PETSc.COMM_SELF)


def _petsc_solve_impl(ksp, b, *, comm=None, print_info=False):
    b_host = np.asarray(b)
    comm = comm or _comm_for_ksp(ksp)

    b_petsc = PETSc.Vec().createSeq(b_host.shape[0], comm=comm)
    b_petsc.getArray(readonly=False)[...] = b_host
    x_petsc = b_petsc.duplicate()
    x_petsc.set(0.0)

    ksp.solve(b_petsc, x_petsc)
    x = np.asarray(x_petsc.getArray()).copy()

    if print_info:
        print("PETSc converged reason:", ksp.getConvergedReason())
        print("PETSc iterations:", ksp.getIterationNumber())
        print("PETSc residual norm:", ksp.getResidualNorm())

    x_petsc.destroy()
    b_petsc.destroy()

    return x


def callPETScCallingJax(ksp, b, *, comm=None, print_info=False):
    """Solve A x = b with an already-configured PETSc KSP from inside JAX.

    The KSP is intentionally closed over as a Python object; only `b` crosses
    the JAX callback boundary as an array. This is the JAX -> PETSc -> JAX path
    when `ksp` uses a PETSc Python Mat whose `mult` calls a JAX function.
    """
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)

    def callback(b_value):
        return _petsc_solve_impl(
            ksp,
            b_value,
            comm=comm,
            print_info=print_info,
        )

    return jax.pure_callback(callback, result_info, b)


def solve(ksp, b, *, comm=None, print_info=False):
    return callPETScCallingJax(ksp, b, comm=comm, print_info=print_info)
