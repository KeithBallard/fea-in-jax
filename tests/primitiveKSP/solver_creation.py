from buildKSP import __CupyCtx as CupyKSPCtx
from buildKSP import __petsc_cleanup as _petsc_cleanup
from buildKSP import __petsc_init as petsc_init
from buildKSP import __retrieve_object as retrieve_object


def petsc_cleanup(solver_or_handle):
    handle = getattr(solver_or_handle, "handle", solver_or_handle)
    return _petsc_cleanup(handle)


__all__ = [
    "CupyKSPCtx",
    "petsc_cleanup",
    "petsc_init",
    "retrieve_object",
]
