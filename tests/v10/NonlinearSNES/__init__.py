"""SNES-focused PETSc/JAX nonlinear helpers."""

from .mat_function_converters import (
    assignPETScMatFromCOOData,
    assignPETScMatFromCOODataDirect,
    convertJAXCOOFuncToPETScMatFunc,
    convertJAXMatFuncToPETScMatFunc,
    convertJAXMatToCOOData,
    convertJAXMatValuesFuncToPETScMatFunc,
    convertJaxMatToCOOData,
)
from .vec_function_converters import (
    assignPETScVecFromJAXBuffer,
    assignPETScVecFromJAXDirect,
    convertJAXVecFuncToPETScVecFunc,
    jaxArrayToPETScVec,
    petscVecToJAX,
)
from .direct_vec_function_converters import (
    convertJAXVecFuncToPETScVecFuncDirect,
)
from .direct_mat_function_converters import (
    convertJAXCOOFuncToPETScMatFuncDirect,
    convertJAXCOOFuncToPETScMatFuncDirectFixedPattern,
    convertJAXCOOFuncToPETScMatFuncDirectPatternAware,
    convertJAXMatFuncToPETScMatFuncDirect,
    convertJAXMatFuncToPETScMatFuncDirectFixedPattern,
)
from .differentiable_snes_prototype import (
    DifferentiableSNESHooks,
    differentiablePETScSolvePrototype,
    make_petsc_snes_callbacks,
    petsc_snes_solve_for_prototype,
    pure_jax_linear_solve_for_testing,
    pure_jax_newton_solve_for_testing,
)

__all__ = [
    "assignPETScMatFromCOOData",
    "assignPETScMatFromCOODataDirect",
    "assignPETScVecFromJAXBuffer",
    "assignPETScVecFromJAXDirect",
    "convertJAXCOOFuncToPETScMatFunc",
    "convertJAXCOOFuncToPETScMatFuncDirect",
    "convertJAXCOOFuncToPETScMatFuncDirectFixedPattern",
    "convertJAXCOOFuncToPETScMatFuncDirectPatternAware",
    "convertJAXMatFuncToPETScMatFunc",
    "convertJAXMatFuncToPETScMatFuncDirect",
    "convertJAXMatFuncToPETScMatFuncDirectFixedPattern",
    "convertJAXMatToCOOData",
    "convertJAXMatValuesFuncToPETScMatFunc",
    "convertJAXVecFuncToPETScVecFunc",
    "convertJAXVecFuncToPETScVecFuncDirect",
    "convertJaxMatToCOOData",
    "DifferentiableSNESHooks",
    "differentiablePETScSolvePrototype",
    "jaxArrayToPETScVec",
    "make_petsc_snes_callbacks",
    "petsc_snes_solve_for_prototype",
    "petscVecToJAX",
    "pure_jax_linear_solve_for_testing",
    "pure_jax_newton_solve_for_testing",
]
