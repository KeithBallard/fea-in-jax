"""
Shim delegating conjugate gradient linear solver implementation to JetSCI.
"""
from jetsci.jax_linear.solve_cg import (
    cg_w_info,
    cg_w_info as cg,
    _cg_solve,
    _isolve,
)

__all__ = ["cg", "cg_w_info", "_cg_solve", "_isolve"]
