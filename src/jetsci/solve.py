from __future__ import annotations

from typing import *

import jax.numpy as jnp
import jax

from .options import *
from .lifecycle import *



def differentiable_solve(solver_options: SolverOptions, R: Callable, J_x: Optional[Callable], x_0, *args) -> tuple[jnp.ndarray, SolverOptions]:
    """
    TODO
    """
    if args:
        R_bar = jax.tree_util.Partial(R, *args)
        J_bar = None if J_x is None else jax.tree_util.Partial(J_x, *args)
    else:
        R_bar = jax.tree_util.Partial(R)
        J_bar = None if J_x is None else jax.tree_util.Partial(J_x)


    solver, solver_options = build_solver_with_reuse(
            solver_options,
            R_bar,
            J_bar,
            x_0,
        )

    x_solution = solver.solve_to_jax(x_0)

    return x_solution, solver_options