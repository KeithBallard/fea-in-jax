"""Level-3 methods for PETSc KSP calling JAX matvecs.

This is the opposite direction from `v10.JaxCallsPETSc`: PETSc owns the KSP
iteration and calls a PETSc Python Mat whose `mult` method evaluates JAX code.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import jax
from petsc4py import PETSc

from ..JaxCallsPETSc.options import PETScMethodOptions, PETScPCType
from .jax_mat import JaxMatContext


DEFAULT_KSP_CALL_JAX_OPTIONS = PETScMethodOptions(pc_type=PETScPCType.NONE)


@dataclass(frozen=True)
class JaxMatObjects:
    """Live PETSc Python Mat plus the context that calls JAX."""

    mat: object
    context: JaxMatContext
    shape: tuple[int, int]


def _comm_from(obj, default=PETSc.COMM_WORLD):
    return getattr(obj, "comm", default)


def init_jax_mat(shape, matvec, *, dmplex=None, comm=None) -> JaxMatObjects:
    """Create a PETSc Python Mat whose multiply calls `matvec`."""
    context = JaxMatContext(matvec, dmplex=dmplex)
    comm = comm or (dmplex.comm if dmplex is not None else PETSc.COMM_WORLD)

    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes(shape)
    mat.setType(PETSc.Mat.Type.PYTHON)
    mat.setPythonContext(context)
    mat.setUp()

    return JaxMatObjects(mat=mat, context=context, shape=tuple(shape))


def cleanup_jax_mat(jax_mat: JaxMatObjects):
    """Destroy a PETSc Python Mat."""
    jax_mat.mat.destroy()
    return jax_mat


def init_ksp_for_jax_mat(
    jax_mat: JaxMatObjects,
    options: PETScMethodOptions | None = None,
    *,
    comm=None,
):
    """Create a PETSc KSP that uses a JAX-backed PETSc Python Mat."""
    options = DEFAULT_KSP_CALL_JAX_OPTIONS if options is None else options
    mat = jax_mat.mat
    comm = comm or _comm_from(mat)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(options.ksp_construction_options()[0])
    ksp.setOperators(mat)
    ksp.getPC().setType(options.pc_construction_options()[0])
    ksp.setFromOptions()
    return ksp


def cleanup_ksp(ksp):
    """Destroy a PETSc KSP created for a JAX-backed Mat."""
    ksp.destroy()
    return ksp


def petsc_solve(ksp, b, *, comm=None, print_info=False):
    """Run `ksp.solve` in ordinary Python and return a NumPy array."""
    b_host = np.asarray(b)
    comm = comm or _comm_from(ksp, PETSc.COMM_SELF)

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


def jax_call_petsc_solve(ksp, b, *, comm=None, print_info=False):
    """Call an already-configured PETSc KSP solve from a JAX context."""
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)

    def callback(b_value):
        return petsc_solve(ksp, b_value, comm=comm, print_info=print_info)

    return jax.pure_callback(callback, result_info, b)
