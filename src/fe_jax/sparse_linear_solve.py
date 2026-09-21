from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
from flax import struct

import jetsci
from jetsci.options import (
    JAXLinearSolverType as LinearSolverType,
    JAXPreconditionerType as PreconditionerType,
    NonlinearSolverType,
    PETScLinearSolverType,
    PETScPreconditionerType,
)
from jetsci.jax_linear import (
    linear_solve as jetsci_linear_solve,
    LinearSolverResultInfo,
    JAXOPT_AVAILABLE,
    CUPY_AVAILABLE,
    PYPARDISO_AVAILABLE,
    PYAMGX_AVAILABLE,
)

from .constraint_system import ConstraintSystem
from .sparse_matrix import apply_dirichlet_bcs_lhs

_logger = logging.getLogger(__name__)


class SolverOptionsMeta(type):
    def __call__(
        cls,
        linear_precond_type: jetsci.JAXPreconditionerType | jetsci.PETScPreconditionerType = jetsci.JAXPreconditionerType.NONE,
        linear_solve_type: jetsci.JAXLinearSolverType | jetsci.PETScLinearSolverType = jetsci.JAXLinearSolverType.CG_JAX_SCIPY_W_INFO,
        linear_max_iter: int = 1000,
        linear_relative_tol: float = 1e-14,
        linear_absolute_tol: float = 1e-10,
        nonlinear_max_iter: int = 10,
        nonlinear_relative_tol: float = 1e-10,
        nonlinear_absolute_tol: float = 1e-8,
        nonlinear_solver_type: jetsci.NonlinearSolverType = jetsci.NonlinearSolverType.JAX_NEWTON_RAPHSON,
        solver_key: int | None = None,
        **kwargs,
    ) -> jetsci.SolverOptions:
        return jetsci.SolverOptions(
            nonlinear_solver_type=nonlinear_solver_type,
            linear_precond_type=linear_precond_type,
            linear_solve_type=linear_solve_type,
            nonlinear_max_iter=nonlinear_max_iter,
            nonlinear_relative_tol=nonlinear_relative_tol,
            nonlinear_absolute_tol=nonlinear_absolute_tol,
            linear_max_iter=linear_max_iter,
            linear_relative_tol=linear_relative_tol,
            linear_absolute_tol=linear_absolute_tol,
            solver_key=solver_key,
        )

    def __instancecheck__(cls, instance):
        return isinstance(instance, jetsci.SolverOptions)


class SolverOptions(metaclass=SolverOptionsMeta):
    """
    Backward-compatible SolverOptions factory returning jetsci.SolverOptions.
    """
    pass


@struct.dataclass
class SolverResultInfo:
    nonlinear_iterations: int
    cumulative_linear_iterations: int
    linear_iterations_per_nonlinear_iteration: jnp.ndarray
    cumulative_residual_norm_history: jnp.ndarray

    def increment_nl_iteration(self):
        """
        Call at the end of each nonlinear iteration to create a copy of the struct that carries
        the solver history forward.
        """
        return SolverResultInfo(
            nonlinear_iterations=self.nonlinear_iterations + 1,
            cumulative_linear_iterations=self.cumulative_linear_iterations,
            linear_iterations_per_nonlinear_iteration=self.linear_iterations_per_nonlinear_iteration,
            cumulative_residual_norm_history=self.cumulative_residual_norm_history,
        )


def init_solver_info(opts: jetsci.SolverOptions) -> SolverResultInfo:
    """
    Initialize a SolverResultInfo object to begin tracking solves.
    """
    return SolverResultInfo(
        nonlinear_iterations=0,
        cumulative_linear_iterations=0,
        linear_iterations_per_nonlinear_iteration=jnp.zeros((opts.nonlinear_max_iter,)),
        cumulative_residual_norm_history=jnp.zeros(
            (opts.linear_max_iter * opts.nonlinear_max_iter + 1,)
        ),
    )


@struct.dataclass
class Residual:
    function: Callable[[jax.Array], jax.Array]
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@struct.dataclass
class Jacobian:
    function: Callable[[jax.Array], jsparse.COO]
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@struct.dataclass
class JacobianDiagonl:
    function: Callable[[jax.Array], jax.Array]
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@partial(jax.jit, static_argnames=["solver_options", "check_consistency"])
def linear_solve(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonl],
    constraints: ConstraintSystem,
    solver_options: jetsci.SolverOptions,
    solver_info_0: SolverResultInfo,
    check_consistency: bool,
    x_0: jnp.ndarray,
    f_ext=None,
    *args,
    **kwargs,
) -> tuple[jnp.ndarray, SolverResultInfo]:
    """
    Solve a linear system of equations emerging from Newton's method: J(x) * dx = -R(x)
    Delegates to JetSCI's linear_solve while applying constraint pre/post processing.
    """
    if residual.dirichlet_bcs_builtin:
        R_w_dirichlet = lambda x: residual.function(x, *args, **kwargs)
    else:
        raise NotImplementedError("Dirichlet BCs must be builtin to residual")

    if jacobian is not None:
        if jacobian.dirichlet_bcs_builtin:
            J_w_dirichlet = lambda x: jacobian.function(x, *args, **kwargs)
        else:
            J_w_dirichlet = lambda x: apply_dirichlet_bcs_lhs(
                jacobian.function(x, *args, **kwargs), constraints.dep_dofs
            )
    else:
        J_w_dirichlet = None

    if jacobian_diagonal is not None:
        if jacobian_diagonal.dirichlet_bcs_builtin:
            diag_J_w_dirichlet = lambda x: jacobian_diagonal.function(
                x, *args, **kwargs
            )
        else:
            diag_J_w_dirichlet = (
                lambda x: jacobian_diagonal.function(x, *args, **kwargs)
                .at[constraints.dep_dofs]
                .set(1.0)
            )
    else:
        diag_J_w_dirichlet = None

    J_vp = jax.tree_util.Partial(
        lambda x, z: jax.jvp(
            R_w_dirichlet,
            (x,),
            (z,),
        )[1],
        x_0,
    )

    if check_consistency:
        v = jax.random.uniform(jax.random.key(0), x_0.shape, x_0.dtype)
        J_dense = jax.jacfwd(R_w_dirichlet)(x_0)

        jax.debug.print(
            "Jacobian-vector product via autodiff matches product via dense Jacobian from jacfwd: {}",
            jnp.isclose(J_vp(v), J_dense @ v).all(),
        )

        if J_w_dirichlet is not None:
            jax.debug.print(
                "Jacobian inferred from residual (via jacfwd) matches given Jacobian function: {}",
                jnp.isclose(J_dense, J_w_dirichlet(x_0).todense()).all(),
            )

        if diag_J_w_dirichlet is not None:
            jax.debug.print(
                "Jacobian diagonal inferred from residual (via diag(jacfwd)) matches Jacobian diagonal function: {}",
                jnp.isclose(jnp.diag(J_dense), diag_J_w_dirichlet(x_0)).all(),
            )

        if J_w_dirichlet is not None:
            jax.debug.print(
                "Jacobian-vector product via autodiff matches product via the given Jacobian function: {}",
                jnp.isclose(J_vp(v), J_w_dirichlet(x_0) @ v).all(),
            )

    R_0 = R_w_dirichlet(x_0)
    b = -R_0

    if J_w_dirichlet is not None:
        A = J_w_dirichlet(x_0)
    else:
        A = J_vp

    custom_precond = None
    if solver_options.linear_precond_type == jetsci.JAXPreconditionerType.JACOBI and diag_J_w_dirichlet is not None:
        d = diag_J_w_dirichlet(x_0)
        custom_precond = lambda r: r / jnp.where(jnp.abs(d) < 1e-15, 1.0, d)

    delta_x, lin_info = jetsci_linear_solve(
        A,
        b,
        solver_options=solver_options,
        preconditioner=custom_precond,
    )

    delta_x = constraints.apply_to_delta_solution(delta_x, x_0)

    k = lin_info.iterations
    hist = lin_info.residual_norm_history
    info = solver_info_0
    if hist is not None:
        cum_hist = lax.dynamic_update_slice(
            info.cumulative_residual_norm_history,
            hist,
            (info.cumulative_linear_iterations + info.nonlinear_iterations,),
        )
    else:
        cum_hist = info.cumulative_residual_norm_history

    info = SolverResultInfo(
        nonlinear_iterations=info.nonlinear_iterations,
        cumulative_linear_iterations=info.cumulative_linear_iterations + k,
        linear_iterations_per_nonlinear_iteration=info.linear_iterations_per_nonlinear_iteration.at[
            info.nonlinear_iterations
        ].set(k),
        cumulative_residual_norm_history=cum_hist,
    )

    return delta_x, info


def plot_solver_info(opts: jetsci.SolverOptions, info: SolverResultInfo):
    """
    Plot residual norm history during iterations.
    """
    import matplotlib.pyplot as plt

    cum_iters = np.concat(
        [
            [0],
            np.cumsum(np.asarray(info.linear_iterations_per_nonlinear_iteration)),
        ]
    )
    x_iter = np.hstack(
        [
            np.linspace(
                cum_iters[i],
                cum_iters[i + 1],
                int(info.linear_iterations_per_nonlinear_iteration[i]) + 1,
            )
            for i in range(info.nonlinear_iterations)
        ]
    )

    y_r_norm = info.cumulative_residual_norm_history[
        0 : info.cumulative_linear_iterations + info.nonlinear_iterations
    ]

    plt.plot(x_iter, y_r_norm)
    plt.title(f"Residual History During Iteration\nUsing {opts.linear_solve_type}")
    plt.xlabel("iteration")
    plt.ylabel("|R|")
    plt.yscale("log")

    for i in range(info.nonlinear_iterations):
        plt.axvline(
            x=cum_iters[i],
            color="r",
            linestyle="--",
            label=f"Start of nonlinear iter {i}",
        )
    plt.legend()
    plt.show()
    plt.savefig("solver_convergence.png")
