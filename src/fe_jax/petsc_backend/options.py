"""PETSc-specific option enums for the Nathan/contact bridge prototype."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PETScMatrixType(Enum):
    """PETSc matrix storage choices for this prototype backend."""

    AIJ = auto()
    AIJCUSPARSE = auto()
    MPIAIJCUSPARSE = auto()


class PETScKSPType(Enum):
    """PETSc KSP choices, mirroring fea-in-jax's enum-style solver options."""

    CG = auto()
    GMRES = auto()
    LGMRES = auto()
    BICGSTAB = auto()
    MINRES = auto()


class PETScPCType(Enum):
    """PETSc preconditioner choices, mirroring fea-in-jax option enums."""

    NONE = auto()
    JACOBI = auto()
    ILU = auto()


_PETSC_MATRIX_TYPES = {
    PETScMatrixType.AIJ: "aij",
    PETScMatrixType.AIJCUSPARSE: "aijcusparse",
    PETScMatrixType.MPIAIJCUSPARSE: "mpiaijcusparse",
}

_PETSC_KSP_TYPES = {
    PETScKSPType.CG: "cg",
    PETScKSPType.GMRES: "gmres",
    PETScKSPType.LGMRES: "lgmres",
    PETScKSPType.BICGSTAB: "bcgs",
    PETScKSPType.MINRES: "minres",
}

_PETSC_PC_TYPES = {
    PETScPCType.NONE: "none",
    PETScPCType.JACOBI: "jacobi",
    PETScPCType.ILU: "ilu",
}


def _coerce_enum(value, enum_type):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in enum_type.__members__:
            return enum_type[normalized]
        petsc_names = {
            PETScMatrixType: _PETSC_MATRIX_TYPES,
            PETScKSPType: _PETSC_KSP_TYPES,
            PETScPCType: _PETSC_PC_TYPES,
        }[enum_type]
        for enum_value, petsc_name in petsc_names.items():
            if value.lower() == petsc_name:
                return enum_value
    raise ValueError(f"Expected {enum_type.__name__}, got {value!r}")


@dataclass(frozen=True)
class PETScKSPOptions:
    """Small prototype option object.

    A real fea-in-jax port should probably fold this into SolverOptions as a
    nested PETSc-specific options dataclass.
    """

    mat_type: PETScMatrixType = PETScMatrixType.AIJCUSPARSE
    ksp_type: PETScKSPType = PETScKSPType.LGMRES
    pc_type: PETScPCType = PETScPCType.JACOBI

    def __post_init__(self):
        object.__setattr__(self, "mat_type", _coerce_enum(self.mat_type, PETScMatrixType))
        object.__setattr__(self, "ksp_type", _coerce_enum(self.ksp_type, PETScKSPType))
        object.__setattr__(self, "pc_type", _coerce_enum(self.pc_type, PETScPCType))

    def as_v7_construction_options(self):
        """Return the full option list expected by the original v7 prototype."""
        return (
            _PETSC_MATRIX_TYPES[self.mat_type],
            _PETSC_KSP_TYPES[self.ksp_type],
            _PETSC_PC_TYPES[self.pc_type],
        )

    def as_matrix_construction_options(self):
        return (_PETSC_MATRIX_TYPES[self.mat_type],)

    def as_pc_construction_options(self):
        return (_PETSC_PC_TYPES[self.pc_type],)

    def as_ksp_construction_options(self):
        return (_PETSC_KSP_TYPES[self.ksp_type],)
