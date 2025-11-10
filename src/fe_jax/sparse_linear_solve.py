import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.extend as jextend
from jax.experimental.buffer_callback import buffer_callback
from jax.dlpack import from_dlpack

from jaxopt import linear_solve as jaxopt_linear_solve

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# For GPU solvers
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cplinalg

from flax import struct
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Optional
from functools import partial

from .utils import debug_print
from .sparse_matrix import *
from .solve_cg import cg as cg_w_info


class LinearSolverType(Enum):
    DIRECT_SPARSE_SOLVE_JNP = (0,)
    DIRECT_INVERSE_JNP = 5
    DIRECT_INVERSE_JAXOPT = 6
    CG_JAXOPT = 10
    CG_SCIPY = 11
    CG_SCIPY_W_INFO = 12
    CG_JACOBI_SCIPY = 13
    CG_JACOBI_SCIPY_W_INFO = 14
    CG_ILU_SCIPY = 15
    CG_ILU_SCIPY_W_INFO = 16
    GMRES_JAXOPT = 20
    GMRES_SCIPY = 21
    BICGSTAB_JAXOPT = 30
    BICGSTAB_SCIPY = 31
    CHOLESKY_JAXOPT = 40
    LU_JAXOPT = 50
    LU_CUPY = 51


@dataclass(eq=True, frozen=True)
class SolverOptions:
    linear_solve_type: LinearSolverType = LinearSolverType.DIRECT_INVERSE_JNP
    linear_max_iter: int = 1000
    linear_relative_tol: float = 1e-14
    linear_absolute_tol: float = 1e-10
    nonlinear_max_iter: int = 10
    nonlinear_relative_tol: float = 1e-10
    nonlinear_absolute_tol: float = 1e-8


@struct.dataclass
class SolverResultInfo:
    nonlinear_iterations: int
    cumulative_linear_iterations: int
    linear_iterations_per_nonlinear_iteration: jnp.ndarray
    # NOTE length will be nonlinear_iterations + cumulative_linear_iterations because the residual
    # norm history for each nonlinear iteration begins with the starting residual norm before a
    # linear solve
    cumulative_residual_norm_history: jnp.ndarray

    def increment_nl_iteration(self):
        return SolverResultInfo(
            nonlinear_iterations=self.nonlinear_iterations + 1,
            cumulative_linear_iterations=self.cumulative_linear_iterations,
            linear_iterations_per_nonlinear_iteration=self.linear_iterations_per_nonlinear_iteration,
            cumulative_residual_norm_history=self.cumulative_residual_norm_history,
        )


def init_solver_info(opts: SolverOptions):
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
    # Function that produces the residual vector (jnp.ndarray).
    # NOTE the solution must be the first argument, though additional args can follow.
    function: Callable[[jax.Array], jax.Array]
    # Indicates whether Dirichlet boundary conditions are built into the residual.
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@struct.dataclass
class Jacobian:
    # Function that produces the sparse matrix (jsparse.COO).
    # NOTE the solution must be the first argument, though additional args can follow.
    function: Callable[[jax.Array], jsparse.COO]
    # Indicates whether Dirichlet boundary conditions are built into the Jacobian.
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@struct.dataclass
class JacobianDiagonl:
    # Function that produces the diagonal of the Jacobian matrix (jnp.ndarray).
    # NOTE the solution must be the first argument, though additional args can follow.
    function: Callable[[jax.Array], jax.Array]
    # Indicates whether Dirichlet boundary conditions are built into the Jacobian.
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@partial(jax.jit, static_argnames=["solver_options"])
def linear_solve(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonl],
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
    solver_options: SolverOptions,
    solver_info_0: SolverResultInfo,
    x_0: jnp.ndarray,
    *args,
    **kwargs,
) -> tuple[jnp.ndarray, SolverResultInfo]:
    """
    TODO document
    """

    if residual.dirichlet_bcs_builtin:
        R_w_dirichlet = lambda x: residual.function(x, *args, **kwargs)
    else:
        raise Exception("TODO (straightforward) implementation needed")

    if jacobian is not None:
        if not jacobian.dirichlet_bcs_builtin:
            J_w_dirichlet = lambda x: apply_dirichlet_bcs_lhs(
                jacobian.function(x, *args, **kwargs), dirichlet_dofs
            )
        else:
            J_w_dirichlet = lambda x: jacobian.function(x, *args, **kwargs)
    else:
        J_w_dirichlet = None

    if jacobian_diagonal is not None:
        if not jacobian_diagonal.dirichlet_bcs_builtin:
            diag_J_w_dirichlet = lambda x: jacobian_diagonal.function(
                x, *args, **kwargs
            )
        else:
            diag_J_w_dirichlet = (
                lambda x: jacobian_diagonal.function(x, *args, **kwargs)
                .at[dirichlet_dofs]
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

    # TODO consider adding an optional check to ensure the provided jacobian function is consistent
    #      with the corrolary via autodiff (though test will of course form the dense matrix)

    R_0 = R_w_dirichlet(x_0)
    info = solver_info_0

    match solver_options.linear_solve_type:

        case LinearSolverType.DIRECT_SPARSE_SOLVE_JNP:
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            delta_x = spsolve(J_sparse, -R_0)

        case LinearSolverType.DIRECT_INVERSE_JNP:
            # NOTE jacfwd of R_w_dirichlet will automatically include in-place elimination
            #      of Dirichlet BCs.
            J_dense = jax.jacfwd(R_w_dirichlet)(x_0)
            delta_x = jnp.array(jnp.dot(jnp.linalg.inv(J_dense), -R_0))

        case LinearSolverType.DIRECT_INVERSE_JAXOPT:
            delta_x = jaxopt_linear_solve.solve_inv(matvec=J_vp, b=-R_0)

        case LinearSolverType.CG_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.CG_SCIPY_W_INFO:
            delta_x, cg_info = cg_w_info(
                A=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                maxiter=solver_options.linear_max_iter,
            )
            info = SolverResultInfo(
                nonlinear_iterations=solver_info_0.nonlinear_iterations,
                cumulative_linear_iterations=solver_info_0.cumulative_linear_iterations
                + cg_info["iterations"],
                linear_iterations_per_nonlinear_iteration=solver_info_0.linear_iterations_per_nonlinear_iteration.at[
                    solver_info_0.nonlinear_iterations
                ].set(
                    cg_info["iterations"]
                ),
                cumulative_residual_norm_history=jax.lax.dynamic_update_slice(
                    operand=solver_info_0.cumulative_residual_norm_history,
                    update=cg_info["residual_norm_history"],
                    start_indices=[
                        solver_info_0.cumulative_linear_iterations
                        + solver_info_0.nonlinear_iterations
                    ],
                ),
            )

        case LinearSolverType.CG_JACOBI_SCIPY:
            assert (
                diag_J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian_diagonal` argument to be provided."

            M_inv_diag = 1.0 / diag_J_w_dirichlet(x_0)

            def jacobi_preconditioner(x):
                return M_inv_diag * x

            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=J_vp,
                M=jacobi_preconditioner,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.CG_JACOBI_SCIPY_W_INFO:
            assert (
                diag_J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian_diagonal` argument to be provided."

            M_inv_diag = 1.0 / diag_J_w_dirichlet(x_0)

            def jacobi_preconditioner(x):
                return M_inv_diag * x

            delta_x, cg_info = cg_w_info(
                A=J_vp,
                M=jacobi_preconditioner,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                maxiter=solver_options.linear_max_iter,
            )
            info = SolverResultInfo(
                nonlinear_iterations=solver_info_0.nonlinear_iterations,
                cumulative_linear_iterations=solver_info_0.cumulative_linear_iterations
                + cg_info["iterations"],
                linear_iterations_per_nonlinear_iteration=solver_info_0.linear_iterations_per_nonlinear_iteration.at[
                    solver_info_0.nonlinear_iterations
                ].set(
                    cg_info["iterations"]
                ),
                cumulative_residual_norm_history=jax.lax.dynamic_update_slice(
                    operand=solver_info_0.cumulative_residual_norm_history,
                    update=cg_info["residual_norm_history"],
                    start_indices=[
                        solver_info_0.cumulative_linear_iterations
                        + solver_info_0.nonlinear_iterations
                    ],
                ),
            )

        case LinearSolverType.CG_ILU_SCIPY_W_INFO:
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)

            ilu_ctx = cupy_spilu_init(J_sparse)

            def ilu_preconditioner(x):
                return cupy_solve(ilu_ctx, x)

            delta_x, cg_info = cg_w_info(
                A=J_vp,
                M=ilu_preconditioner,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                maxiter=solver_options.linear_max_iter,
            )
            info = SolverResultInfo(
                nonlinear_iterations=solver_info_0.nonlinear_iterations,
                cumulative_linear_iterations=solver_info_0.cumulative_linear_iterations
                + cg_info["iterations"],
                linear_iterations_per_nonlinear_iteration=solver_info_0.linear_iterations_per_nonlinear_iteration.at[
                    solver_info_0.nonlinear_iterations
                ].set(
                    cg_info["iterations"]
                ),
                cumulative_residual_norm_history=jax.lax.dynamic_update_slice(
                    operand=solver_info_0.cumulative_residual_norm_history,
                    update=cg_info["residual_norm_history"],
                    start_indices=[
                        solver_info_0.cumulative_linear_iterations
                        + solver_info_0.nonlinear_iterations
                    ],
                ),
            )

        case LinearSolverType.CG_JAXOPT:
            delta_x = jaxopt_linear_solve.solve_cg(
                matvec=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.GMRES_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.gmres(
                A=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.GMRES_JAXOPT:
            delta_x = jaxopt_linear_solve.solve_gmres(
                matvec=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.BICGSTAB_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.bicgstab(
                A=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.BICGSTAB_JAXOPT:
            delta_x = jaxopt_linear_solve.solve_bicgstab(
                matvec=J_vp,
                b=-R_0,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.CHOLESKY_JAXOPT:
            delta_x = jaxopt_linear_solve.solve_cholesky(matvec=J_vp, b=-R_0)

        case LinearSolverType.LU_JAXOPT:
            delta_x = jaxopt_linear_solve.solve_lu(matvec=J_vp, b=-R_0)

        case LinearSolverType.LU_CUPY:
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)

            ilu_ctx = cupy_splu_init(J_sparse)
            delta_x = cupy_solve(ilu_ctx, -R_0)

        # Note: consider implementing spai preconditioner
        # https://tbetcke.github.io/hpc_lecture_notes/it_solvers4.html

        case _:
            raise Exception(
                f"Linear solver type {solver_options.linear_solve_type} is not implemented"
            )

    # jax.scipy solvers will not arrive at the right values for the constraints for any size of
    # problem but even the jaxopt solvers will only get close for large problems.
    # Consequently, overwrite the values directly to ensure the BCs are right, even though the
    # residual may increase.
    delta_x = delta_x.at[dirichlet_dofs].set(dirichlet_values - x_0[dirichlet_dofs])

    return delta_x, info


def plot_solver_info(opts: SolverOptions, info: SolverResultInfo):
    """
    TODO document
    """
    import matplotlib.pyplot as plt

    x_iter = jnp.linspace(
        0,
        info.cumulative_linear_iterations,
        info.cumulative_linear_iterations + 1,
        dtype=jnp.int32,
    )
    y_r_norm = info.cumulative_residual_norm_history[
        0 : info.cumulative_linear_iterations + 1
    ]

    plt.plot(x_iter, y_r_norm)
    plt.title(f"Residual History During Iteration\nUsing {opts.linear_solve_type}")
    plt.xlabel("iteration")
    plt.ylabel("|R|")
    plt.yscale("log")

    cum_iters = np.concat(
        [
            [0],
            np.cumsum(np.asarray(info.linear_iterations_per_nonlinear_iteration)),
        ]
    )
    for i in range(info.nonlinear_iterations + 1):
        plt.axvline(
            x=cum_iters[i],
            color="r",
            linestyle="--",
            label=f"Start of nonlinear iter {i}",
        )
    plt.legend()

    plt.show()
    plt.savefig("solver_convergence.png")


def __solve_cpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a CPU backend.
    Returns the solution, x.
    """
    A_jax_csr = coo_to_csr(A)
    A_csr = scipy.sparse.csr_matrix(
        (
            np.array(A_jax_csr.data),
            np.array(A_jax_csr.indices),
            np.array(A_jax_csr.indptr),
        ),
        shape=(A.shape[0], A.shape[1]),
    )
    return scipy.sparse.linalg.spsolve(A_csr, b)


@jax.jit
def cupy_spsolve(A: jsparse.CSR, b: jnp.ndarray):

    def kernel(ctx, out, A: jsparse.CSR, b):
        A_cp = cpsparse.csr_matrix(
            (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_cp.has_canonical_format = True
        # cp.savetxt("A_cp.csv", A_cp.todense())
        cp.asarray(out)[...] = cplinalg.spsolve(A_cp, cp.asarray(b))

    out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
    cupy_callback = buffer_callback(kernel, out_type)
    return cupy_callback(A, b)


@jax.jit
def __solve_gpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a GPU backend.
    Returns the solution, x.
    """
    A_csr = coo_to_csr(A)
    return cupy_spsolve(A_csr, b)


def spsolve(A: jsparse.COO, b: jnp.ndarray) -> jnp.ndarray:
    """
    Sparse direct solve for system A*x = b.
    Returns the solution, x.
    """
    match jextend.backend.get_backend().platform:
        case "cpu":
            return jnp.array(__solve_cpu(A, b))
        case "gpu":
            return __solve_gpu(A, b)
    raise Exception(f"Backend {jextend.backend.get_backend().platform} unsupported.")


from cupyx.scipy.sparse.linalg._solve import CusparseLU

# Global registry to hold generic Python objects
_OBJECT_STORE = {}
_NEXT_ID = 0


def _store_object(obj):
    global _NEXT_ID
    uid = _NEXT_ID
    _OBJECT_STORE[uid] = obj
    _NEXT_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type


def _retrieve_object(uid):
    # Ensure uid is a standard Python int for dict lookup
    return _OBJECT_STORE[int(uid)]


@struct.dataclass
class CupyCtx:
    handle: jnp.ndarray


def _cupy_spilu_init_impl(A: jsparse.CSR):
    A_cp = cpsparse.csr_matrix(
        (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
        shape=A.shape,
    )
    A_cp.has_canonical_format = True
    ilu_obj = cplinalg.spilu(A_cp, fill_factor=1.0)
    return _store_object(ilu_obj)


@jax.jit
def cupy_spilu_init(A: jsparse.COO) -> CupyCtx:
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    handle = jax.pure_callback(_cupy_spilu_init_impl, result_info, coo_to_csr(A))
    jax.debug.print("cupy_ilu_ctx {}", handle)
    return CupyCtx(handle=handle)


def _cupy_splu_init_impl(A: jsparse.CSR):
    A_cp = cpsparse.csr_matrix(
        (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
        shape=A.shape,
    )
    A_cp.has_canonical_format = True
    ilu_obj = cplinalg.splu(A_cp)
    return _store_object(ilu_obj)


@jax.jit
def cupy_splu_init(A: jsparse.COO) -> CupyCtx:
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    handle = jax.pure_callback(_cupy_splu_init_impl, result_info, coo_to_csr(A))
    jax.debug.print("cupy_ilu_ctx {}", handle)
    return CupyCtx(handle=handle)


def _cupy_solve_impl(ctx, out, handle: jnp.ndarray, b: jnp.ndarray):
    # Retrieve the opaque object using the handle
    cupy_obj = _retrieve_object(cp.asarray(handle))
    cp.asarray(out)[...] = cupy_obj.solve(cp.asarray(b))


@jax.jit
def cupy_solve(ctx: CupyCtx, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return buffer_callback(_cupy_solve_impl, result_info)(ctx.handle, b)
