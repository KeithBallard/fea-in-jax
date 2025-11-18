import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.extend as jextend
from jax.experimental.buffer_callback import buffer_callback
from jax.experimental import checkify
from jax.dlpack import from_dlpack

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from flax import struct
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Optional
from functools import partial

from .utils import debug_print
from .sparse_matrix import *
from .solve_cg import cg as cg_w_info

try:
    import jaxopt.linear_solve

    JAXOPT_AVAILABLE = True
    print("'jaxopt' imported, adding related solvers.")
except ImportError:
    JAXOPT_AVAILABLE = False
    print("'jaxopt' was not imported successfully, skipping related solvers.")

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cplinalg

    CUPY_AVAILABLE = True
    print("'cupy' imported, adding related solvers/preconditioners.")
except ImportError:
    CUPY_AVAILABLE = False
    print(
        "'cupy' was not imported successfully, skipping related solvers/preconditioners."
    )


try:
    import pyamgx

    assert (
        CUPY_AVAILABLE
    ), "The interface to `pyamgx` requires `cupy`, which was not available. Please install `cupy`."
    PYAMX_AVAILABLE = True
    print("'pyamgx' imported, adding related solvers/preconditioners.")
except ImportError:
    PYAMX_AVAILABLE = False
    print(
        "'pyamgx' was not imported successfully, skipping related solvers/preconditioners."
    )


try:
    import pypardiso

    assert (
        CUPY_AVAILABLE
    ), "The interface to `pypardiso` requires `cupy`, which was not available. Please install `cupy`."
    PYPARDISO_AVAILABLE = True
    print("'pypardiso' imported, adding related solvers.")
except ImportError:
    PYPARDISO_AVAILABLE = False
    print("'pypardiso' was not imported successfully, skipping related solvers.")


class PreconditionerType(Enum):
    NONE = auto()
    JACOBI = auto()
    ILU_CUPY = auto()


class LinearSolverType(Enum):
    DENSE_INVERSE_JNP = auto()
    CG_JAX_SCIPY = auto()
    CG_JAX_SCIPY_W_INFO = auto()
    GMRES_JAX_SCIPY = auto()
    BICGSTAB_JAX_SCIPY = auto()
    DENSE_INVERSE_JAXOPT = auto()
    LU_JAXOPT = auto()
    CHOLESKY_JAXOPT = auto()
    CG_JAXOPT = auto()
    GMRES_JAXOPT = auto()
    BICGSTAB_JAXOPT = auto()
    SPSOLVE_CUPY = auto()
    LU_CUPY = auto()
    # TODO GMRES_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.gmres.html
    # TODO CGS_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.cgs.html
    # TODO MINRES_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.minres.html
    AMGX = auto()
    SPSOLVE_PYPARDISO = auto()


@dataclass(eq=True, frozen=True)
class SolverOptions:
    linear_precond_type: PreconditionerType = PreconditionerType.NONE
    linear_solve_type: LinearSolverType = LinearSolverType.CG_JAX_SCIPY_W_INFO
    linear_max_iter: int = 1000
    linear_relative_tol: float = 1e-14
    linear_absolute_tol: float = 1e-10
    nonlinear_max_iter: int = 10
    nonlinear_relative_tol: float = 1e-10
    nonlinear_absolute_tol: float = 1e-8

    def __post_init__(self):
        # Validate that the selected preconditioner is available
        if "CUPY" in self.linear_precond_type.name:
            assert (
                CUPY_AVAILABLE
            ), f"The selected preconditioner ({self.linear_precond_type.name}) requires cupy, which is not available."
        # Validate that the selected solver is available
        if "JAXOPT" in self.linear_solve_type.name:
            assert (
                JAXOPT_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires jaxopt, which is not available."
        if "CUPY" in self.linear_solve_type.name:
            assert (
                CUPY_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires cupy, which is not available."
        if "AMGX" in self.linear_solve_type.name:
            assert (
                PYAMX_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires pyamgx, which is not available."
        if "PYPARDISO" in self.linear_solve_type.name:
            assert (
                PYPARDISO_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires pypardiso, which is not available."


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


def init_solver_info(opts: SolverOptions):
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


@partial(jax.jit, static_argnames=["solver_options", "check_consistency"])
def linear_solve(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonl],
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
    solver_options: SolverOptions,
    solver_info_0: SolverResultInfo,
    check_consistency: bool,
    x_0: jnp.ndarray,
    *args,
    **kwargs,
) -> tuple[jnp.ndarray, SolverResultInfo]:
    """
    Solve a linear system of equations emerging from Newton's method: J(x) * x = -R(x)

    TODO finish documentation
    """

    if residual.dirichlet_bcs_builtin:
        R_w_dirichlet = lambda x: residual.function(x, *args, **kwargs)
    else:
        raise Exception("TODO (straightforward) implementation needed")

    if jacobian is not None:
        if jacobian.dirichlet_bcs_builtin:
            J_w_dirichlet = lambda x: jacobian.function(x, *args, **kwargs)
        else:
            J_w_dirichlet = lambda x: apply_dirichlet_bcs_lhs(
                jacobian.function(x, *args, **kwargs), dirichlet_dofs
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
    delta_x = (
        jnp.zeros_like(R_0)
        .at[dirichlet_dofs]
        .set(dirichlet_values - x_0[dirichlet_dofs])
    )
    info = solver_info_0

    match solver_options.linear_precond_type:
        ##########################################################################################
        # jax native preconditioners
        case PreconditionerType.NONE:
            preconditioner = None

        case PreconditionerType.JACOBI:
            assert (
                diag_J_w_dirichlet is not None
            ), f"{solver_options.linear_precond_type} requires the `jacobian_diagonal` argument to be provided."

            preconditioner = lambda x: x / diag_J_w_dirichlet(x_0)

        ##########################################################################################
        # cupy preconditioners

        case PreconditionerType.ILU_CUPY:
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            ilu_ctx = __cupy_spilu_init(J_sparse)
            preconditioner = lambda x: __cupy_solve(ilu_ctx, x)

        case _:
            raise Exception(
                f"Preconditioner type {solver_options.linear_precond_type} is not implemented"
            )

    match solver_options.linear_solve_type:
        ##########################################################################################
        # jax native solvers

        case LinearSolverType.DENSE_INVERSE_JNP:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            # NOTE jacfwd of R_w_dirichlet will automatically include in-place elimination
            #      of Dirichlet BCs.
            J_dense = jax.jacfwd(R_w_dirichlet)(x_0)
            delta_x = jnp.array(jnp.dot(jnp.linalg.inv(J_dense), -R_0))

        case LinearSolverType.CG_JAX_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.CG_JAX_SCIPY_W_INFO:
            delta_x, cg_info = cg_w_info(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
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

        case LinearSolverType.GMRES_JAX_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.gmres(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.BICGSTAB_JAX_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.bicgstab(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        ##########################################################################################
        # jaxopt solvers

        case LinearSolverType.DENSE_INVERSE_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_inv(matvec=J_vp, b=-R_0)  # type: ignore

        case LinearSolverType.LU_JAXOPT:
            delta_x = jaxopt.linear_solve.solve_lu(matvec=J_vp, b=-R_0)  # type: ignore

        case LinearSolverType.CHOLESKY_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_cholesky(matvec=J_vp, b=-R_0)  # type: ignore

        case LinearSolverType.CG_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_cg(  # type: ignore
                matvec=J_vp,
                b=-R_0,
                init=delta_x,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.GMRES_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_gmres(  # type: ignore
                matvec=J_vp,
                b=-R_0,
                init=delta_x,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.BICGSTAB_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_bicgstab(  # type: ignore
                matvec=J_vp,
                b=-R_0,
                init=delta_x,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        ##########################################################################################
        # cupy solvers

        case LinearSolverType.SPSOLVE_CUPY:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            delta_x = __spsolve(J_sparse, -R_0)

        case LinearSolverType.LU_CUPY:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)

            ilu_ctx = __cupy_splu_init(J_sparse)
            delta_x = __cupy_solve(ilu_ctx, -R_0)

        ##########################################################################################
        # amgx solvers

        case LinearSolverType.AMGX:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            ctx = __amgx_init(J_sparse)
            delta_x = __amgx_solve(ctx, -R_0, delta_x)
            __amgx_finalize(ctx)

        ##########################################################################################
        # pypardiso solvers

        case LinearSolverType.SPSOLVE_PYPARDISO:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            delta_x = __pypardiso_solve(J_sparse, -R_0)

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


##################################################################################################
# Object store for external library context information

# Global registry to hold generic Python objects
_OBJECT_STORE = {}
_NEXT_ID = 0


def __store_object(obj):
    global _NEXT_ID
    uid = _NEXT_ID
    _OBJECT_STORE[uid] = obj
    _NEXT_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type


def __retrieve_object(uid):
    return _OBJECT_STORE[int(uid)]


##################################################################################################
# CUPY wrappers

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cplinalg

    @struct.dataclass
    class __SolverCtx:
        handle: jnp.ndarray

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
    def __cupy_spsolve(A: jsparse.CSR, b: jnp.ndarray):

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
        return __cupy_spsolve(A_csr, b)

    def __spsolve(A: jsparse.COO, b: jnp.ndarray) -> jnp.ndarray:
        """
        Sparse direct solve for system A*x = b.
        Returns the solution, x.
        """
        match jextend.backend.get_backend().platform:
            case "cpu":
                return jnp.array(__solve_cpu(A, b))
            case "gpu":
                return __solve_gpu(A, b)
        raise Exception(
            f"Backend {jextend.backend.get_backend().platform} unsupported."
        )

    def __cupy_spilu_init_impl(A: jsparse.CSR):
        A_cp = cpsparse.csr_matrix(
            (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_cp.has_canonical_format = True
        ilu_obj = cplinalg.spilu(A_cp, fill_factor=1.0)
        return __store_object(ilu_obj)

    @jax.jit
    def __cupy_spilu_init(A: jsparse.COO) -> __SolverCtx:
        result_info = jax.ShapeDtypeStruct((), jnp.int64)
        handle = jax.pure_callback(__cupy_spilu_init_impl, result_info, coo_to_csr(A))
        return __SolverCtx(handle=handle)

    def __cupy_splu_init_impl(A: jsparse.CSR):
        A_cp = cpsparse.csr_matrix(
            (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_cp.has_canonical_format = True
        ilu_obj = cplinalg.splu(A_cp)
        return __store_object(ilu_obj)

    @jax.jit
    def __cupy_splu_init(A: jsparse.COO) -> __SolverCtx:
        result_info = jax.ShapeDtypeStruct((), jnp.int64)
        handle = jax.pure_callback(__cupy_splu_init_impl, result_info, coo_to_csr(A))
        return __SolverCtx(handle=handle)

    def __cupy_solve_impl(ctx, out, handle: jnp.ndarray, b: jnp.ndarray):
        # Retrieve the opaque object using the handle
        cupy_obj = __retrieve_object(cp.asarray(handle))
        cp.asarray(out)[...] = cupy_obj.solve(cp.asarray(b))

    @jax.jit
    def __cupy_solve(ctx: __SolverCtx, b: jnp.ndarray):
        result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return buffer_callback(__cupy_solve_impl, result_info)(ctx.handle, b)


##################################################################################################
# AMGX wrappers

if PYAMX_AVAILABLE:
    import pyamgx
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse

    pyamgx.initialize()

    # Refer to https://github.com/NVIDIA/AMGX/tree/main/src/configs for examples
    cfg = pyamgx.Config().create_from_dict(
        {
            "config_version": 2,
            "solver": {
                "preconditioner": {"scope": "amg", "solver": "NOSOLVER"},
                "use_scalar_norm": 1,
                "solver": "PCG",
                "print_solve_stats": 1,
                "obtain_timings": 1,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI",
                "scope": "main",
                "max_iters": 100,
                "tolerance": 1e-06,
                "norm": "L2",
            },
        }
    )

    rsc = pyamgx.Resources().create_simple(cfg)

    @struct.dataclass
    class __AMGXCtx:
        cfg_handle: jnp.ndarray
        rsc_handle: jnp.ndarray
        A_handle: jnp.ndarray
        b_handle: jnp.ndarray
        x_handle: jnp.ndarray
        solver_handle: jnp.ndarray

    def __amgx_init_impl(
        A: jsparse.COO,
    ) -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64, np.int64]:
        A_amgx = pyamgx.Matrix().create(rsc)
        A_cp = cpsparse.csr_matrix(
            (
                cp.asarray(A.data),
                (cp.asarray(A.row), cp.asarray(A.col)),
            ),  # cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_amgx.upload_CSR(A_cp)

        b_amgx = pyamgx.Vector().create(rsc)
        x_amgx = pyamgx.Vector().create(rsc)

        solver = pyamgx.Solver().create(rsc, cfg)
        solver.setup(A_amgx)

        return (
            __store_object(cfg),
            __store_object(rsc),
            __store_object(A_amgx),
            __store_object(b_amgx),
            __store_object(x_amgx),
            __store_object(solver),
        )

    @jax.jit
    def __amgx_init(A: jsparse.COO) -> __AMGXCtx:
        result_info = (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
        )
        cfg_handle, rsc_handle, A_handle, b_handle, x_handle, solver_handle = (
            jax.pure_callback(__amgx_init_impl, result_info, A)  # coo_to_csr(A)
        )
        return __AMGXCtx(
            cfg_handle=cfg_handle,
            rsc_handle=rsc_handle,
            A_handle=A_handle,
            b_handle=b_handle,
            x_handle=x_handle,
            solver_handle=solver_handle,
        )

    def __amgx_solve_impl(
        ctx,
        out,
        b_handle: jnp.ndarray,
        x_handle: jnp.ndarray,
        solver_handle: jnp.ndarray,
        b: jnp.ndarray,
        x0: jnp.ndarray,
    ):
        # Retrieve the opaque object using the handle
        b_amgx = __retrieve_object(cp.asarray(b_handle))
        x_amgx = __retrieve_object(cp.asarray(x_handle))
        solver_amgx = __retrieve_object(cp.asarray(solver_handle))

        b_amgx.upload(cp.asarray(b))
        x_amgx.upload(cp.asarray(x0).get())
        print("x_amgx", cp.asarray(x_amgx.download()))
        solver_amgx.solve(b_amgx, x_amgx, zero_initial_guess=False)

        # x = cp.zeros_like(cp.asarray(b))
        cp.asarray(out)[...] = cp.asarray(x_amgx.download())

    @jax.jit
    def __amgx_solve(ctx: __AMGXCtx, b: jnp.ndarray, x0: jnp.ndarray):
        result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return buffer_callback(__amgx_solve_impl, result_info)(
            ctx.x_handle, ctx.b_handle, ctx.solver_handle, b, x0
        )

    def __amgx_finalize_impl(
        A_handle: jnp.ndarray,
        b_handle: jnp.ndarray,
        x_handle: jnp.ndarray,
        solver_handle: jnp.ndarray,
        cfg_handle: jnp.ndarray,
        rsc_handle: jnp.ndarray,
    ):
        import pyamgx

        A_amgx = __retrieve_object(cp.asarray(A_handle))
        b_amgx = __retrieve_object(cp.asarray(b_handle))
        x_amgx = __retrieve_object(cp.asarray(x_handle))
        solver_amgx = __retrieve_object(cp.asarray(solver_handle))
        rsc = __retrieve_object(cp.asarray(rsc_handle))
        cfg = __retrieve_object(cp.asarray(cfg_handle))

        A_amgx.destroy()
        x_amgx.destroy()
        b_amgx.destroy()
        solver_amgx.destroy()
        rsc.destroy()
        cfg.destroy()
        # pyamgx.finalize()

    def __amgx_finalize(ctx: __AMGXCtx):
        jax.pure_callback(
            __amgx_finalize_impl,
            ctx.A_handle,
            ctx.x_handle,
            ctx.b_handle,
            ctx.solver_handle,
            ctx.cfg_handle,
            ctx.rsc_handle,
        )


##################################################################################################
# PYPARDISO wrappers

if PYPARDISO_AVAILABLE:
    import pypardiso
    import cupy as cp

    def __pypardiso_solve_impl(
        #ctx, <- buffer_callback implementation
        #out, <- buffer_callback implementation
        A_data: jnp.ndarray,
        A_row: jnp.ndarray,
        A_col: jnp.ndarray,
        b: jnp.ndarray,
    ):
        A_scipy = scipy.sparse.csr_matrix(
            (
                cp.asarray(A_data).get().astype(np.float64),
                (cp.asarray(A_row).get().astype(np.int32), cp.asarray(A_col).get().astype(np.int32)),
            ),
            shape=(b.shape[0], b.shape[0]),
        )
        b_np = cp.asarray(b).get().astype(np.float64)
        result = pypardiso.spsolve(A_scipy, b_np)
        #cp.asarray(out)[...] = cp.asarray(result) <- buffer_callback implementation
        return result

    @jax.jit
    def __pypardiso_solve(A: jsparse.COO, b: jnp.ndarray):
        """
        # Import Note
        The buffer_callback would theoretically be more efficient, but there is a bug in JAX resulting
        in bizzare values in the arrays.  If jax.debug.print is not called to view the arrays, they become
        uninitiailized values within __pypardiso_solve_impl. It is like the XLA compiler thinks the arrays
        are not used and the computation branch is pruned. Consequently, I will use a pure_callback variant.
        Fortunately, for this solver, the cost is low since the solve is performed on the host anyway.
        """
        result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return jax.pure_callback(__pypardiso_solve_impl, result_info, A.data, A.row, A.col, b)
        #result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        #jax.debug.print("A.row - jax {}", A.row)
        #jax.debug.print("A.data - jax {}", A.data)
        #return buffer_callback(
        #    __pypardiso_solve_impl, result_info, command_buffer_compatible=False
        #)(A.data, A.row, A.col, b)
