from typing import *
from enum import Enum
from enum import auto

from dataclasses import dataclass



class NonlinearSolverType(Enum):
    JAX_NEWTON_RAPHSON = auto()
    PETSC_SNES         = auto()

class PETScPreconditionerType(Enum):
    NONE   = auto()
    JACOBI = auto()
    ILU    = auto()

class JAXPreconditionerType(Enum):
    NONE     = auto()
    JACOBI   = auto()
    ILU_CUPY = auto()

class PETScLinearSolverType(Enum):
    CG         = auto()
    LGMRES     = auto()
    BCGS       = auto()
    PREONLY    = auto()

class JAXLinearSolverType(Enum):
    DENSE_INVERSE_JNP    = auto()
    CG_JAX_SCIPY         = auto()
    CG_JAX_SCIPY_W_INFO  = auto()
    GMRES_JAX_SCIPY      = auto()
    BICGSTAB_JAX_SCIPY   = auto()
    DENSE_INVERSE_JAXOPT = auto()
    LU_JAXOPT            = auto()
    CHOLESKY_JAXOPT      = auto()
    CG_JAXOPT            = auto()
    GMRES_JAXOPT         = auto()
    BICGSTAB_JAXOPT      = auto()
    SPSOLVE_CUPY         = auto()
    LU_CUPY              = auto()
    # TODO GMRES_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.gmres.html
    # TODO CGS_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.cgs.html
    # TODO MINRES_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.minres.html
    AMGX                 = auto()
    SPSOLVE_PYPARDISO    = auto()


@dataclass(eq=True, frozen=True)
class SolverOptions:
    nonlinear_solver_type:  NonlinearSolverType
    linear_precond_type:    PETScPreconditionerType | JAXPreconditionerType
    linear_solve_type:      PETScLinearSolverType | JAXLinearSolverType

    nonlinear_max_iter:     int = 10
    nonlinear_relative_tol: float = 1e-10
    nonlinear_absolute_tol: float = 1e-8
    nonlinear_step_tol:     float = 0.0

    linear_max_iter:        int = 1000
    linear_relative_tol:    float = 1e-14
    linear_absolute_tol:    float = 1e-10

    solver_key:             int | None = None
