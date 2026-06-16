from .solver_creation import CupyKSPCtx
from runKSP import __petsc_solve as petsc_solve
from runKSP import __petsc_solve_transpose as petsc_solve_transpose


def _solver_from_handle(handle):
    return CupyKSPCtx(handle=handle)


def petsc_solve_handle(handle, b):
    return petsc_solve(_solver_from_handle(handle), b)


def petsc_solve_transpose_handle(handle, b):
    return petsc_solve_transpose(_solver_from_handle(handle), b)


__all__ = [
    "petsc_solve",
    "petsc_solve_handle",
    "petsc_solve_transpose",
    "petsc_solve_transpose_handle",
]
