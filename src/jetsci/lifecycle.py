import jax
import jax.numpy as jnp

from . import petsc_snes
from .options import *

def build_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J_x: jax.tree_util.Partial,
    x_0: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, SolverOptions]:
    match options.nonlinear_solver_type:
        case NonlinearSolverType.JAX_NEWTON_RAPHSON:
            return jax_netwon_raph
        case NonlinearSolverType.PETSC_SNES:
            return petsc_snes.build_petsc_solver_with_reuse(
                options,
                R,
                J_x,
                x_0,
            )