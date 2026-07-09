"""SNES construction helpers.

This file is intentionally a thin scaffold for now. The callback conversion
pieces live in the Vec/Mat converter modules; this module will eventually own
the PETSc SNES object lifecycle and attach those callbacks.
"""

from __future__ import annotations

from .direct_mat_function_converters import convertJAXCOOFuncToPETScMatFuncDirectPatternAware
from .direct_vec_function_converters import convertJAXVecFuncToPETScVecFuncDirect


def buildSNES(
    residual_callable,
    jacobian_coo_callable,
    snes_options,
    args=None,
    kwargs=None,
):
    """Build a PETSc SNES object from JAX residual/Jacobian callbacks.

    This is not implemented yet because v10 still needs to settle the public
    SNES lifecycle object and option model. The intended callback conversion is
    captured here so the design has a stable landing spot.
    """
    del snes_options, kwargs
    petsc_residual = convertJAXVecFuncToPETScVecFuncDirect(residual_callable, args)
    petsc_jacobian = convertJAXCOOFuncToPETScMatFuncDirectPatternAware(jacobian_coo_callable, args)
    raise NotImplementedError(
        "SNES object construction is not implemented yet. "
        "Use petsc_residual/petsc_jacobian as the callback conversion pattern."
    )


def adjustSNES(snes, snes_update_options):
    """Update an existing SNES object between runs."""
    raise NotImplementedError("SNES update helpers are not implemented yet")
