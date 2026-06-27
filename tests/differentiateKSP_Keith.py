import jax
import jax.numpy as jnp
from jax.interpreters import ad, batching, mlir

import buildKSP_Keith
import runKSP_Keith


try:
    from jax.extend import core as jax_core
except ImportError:
    from jax import core as jax_core

try:
    from jax import core as legacy_core
except ImportError:
    legacy_core = jax_core


linear_solver_solve_p = jax_core.Primitive("keith_linear_solver_solve")


def _solver_handle(solver_or_handle):
    return getattr(solver_or_handle, "handle", solver_or_handle)


def _solver_from_handle(handle):
    return buildKSP_Keith.__CupyCtx(handle=handle)


def _is_zero(value):
    return type(value) is ad.Zero


def _zero_from_value(value):
    if hasattr(ad.Zero, "from_primal_value"):
        return ad.Zero.from_primal_value(value)
    aval = jax.typeof(value) if hasattr(jax, "typeof") else legacy_core.get_aval(value)
    return ad.Zero(aval)


def _coo_matvec(rows, cols, vals, x):
    return jnp.zeros_like(x).at[rows].add(vals * x[cols])


def _solve_with_handle(handle, pc_handle, b):
    return runKSP_Keith.__petsc_solve(_solver_from_handle(handle), _solver_from_handle(pc_handle), b)


def _solve_transpose_with_handle(handle, pc_handle, b):
    return runKSP_Keith.__petsc_solve_transpose(_solver_from_handle(handle), _solver_from_handle(pc_handle), b)


def _lowered_solve(handle, pc_handle, shape, rows, cols, vals, b):
    del shape, rows, cols, vals
    return _solve_with_handle(handle, pc_handle, b)


def linearSolverSolve(solver, pc, shape, rows, cols, vals, b):
    return linear_solver_solve_p.bind(_solver_handle(solver), _solver_handle(pc), shape, rows, cols, vals, b)


def _linear_solver_solve_impl(handle, pc_handle, shape, rows, cols, vals, b):
    del shape, rows, cols, vals
    return _solve_with_handle(handle, pc_handle, b)


def _linear_solver_solve_abstract_eval(handle_aval, pc_handle_aval, shape_aval, rows_aval, cols_aval, vals_aval, b_aval):
    del handle_aval, pc_handle_aval, shape_aval, rows_aval, cols_aval, vals_aval
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array(b_aval.shape, b_aval.dtype)


def _linear_solver_solve_lowering(ctx, handle, pc_handle, shape, rows, cols, vals, b):
    return mlir.lower_fun(_lowered_solve, multiple_results=False)(ctx, handle, pc_handle, shape, rows, cols, vals, b)


def _linear_solver_solve_jvp(primals, tangents):
    handle, pc_handle, shape, rows, cols, vals, b = primals
    _, _, _, _, _, vals_dot, b_dot = tangents

    x = linear_solver_solve_p.bind(handle, pc_handle, shape, rows, cols, vals, b)
    if _is_zero(vals_dot) and _is_zero(b_dot):
        return x, _zero_from_value(x)

    rhs_dot = jnp.zeros_like(b) if _is_zero(b_dot) else b_dot
    if not _is_zero(vals_dot):
        rhs_dot = rhs_dot - _coo_matvec(rows, cols, vals_dot, x)

    x_dot = linear_solver_solve_p.bind(handle, pc_handle, shape, rows, cols, vals, rhs_dot)
    return x, x_dot


def _linear_solver_solve_transpose(ct, handle, pc_handle, shape, rows, cols, vals, b):
    handle_bar = None
    pc_handle_bar = None
    shape_bar = None
    rows_bar = None
    cols_bar = None

    vals_is_unknown = isinstance(vals, ad.UndefinedPrimal)
    b_is_unknown = isinstance(b, ad.UndefinedPrimal)

    if _is_zero(ct):
        vals_bar = ad.Zero(vals.aval) if vals_is_unknown else None
        b_bar = ad.Zero(b.aval) if b_is_unknown else None
        return handle_bar, pc_handle_bar, shape_bar, rows_bar, cols_bar, vals_bar, b_bar

    if vals_is_unknown:
        raise NotImplementedError(
            "keith_linear_solver_solve transpose needs primal vals to use the live PETSc solver"
        )

    adjoint = _solve_transpose_with_handle(handle, pc_handle, ct)
    if b_is_unknown:
        vals_bar = None
    else:
        x = _solve_with_handle(handle, pc_handle, b)
        vals_bar = -adjoint[rows] * x[cols]

    b_bar = adjoint if b_is_unknown else None
    return handle_bar, pc_handle_bar, shape_bar, rows_bar, cols_bar, vals_bar, b_bar


def _linear_solver_solve_batch(args, batch_dims):
    handle, pc_handle, shape, rows, cols, vals, b = args
    handle_bdim, pc_handle_bdim, shape_bdim, rows_bdim, cols_bdim, vals_bdim, b_bdim = batch_dims

    if handle_bdim is not None:
        raise NotImplementedError("Batching over PETSc solver handles is not supported")
    if pc_handle_bdim is not None:
        raise NotImplementedError("Batching over PETSc preconditioner handles is not supported")
    if shape_bdim is not None or rows_bdim is not None or cols_bdim is not None:
        raise NotImplementedError("Batching over COO metadata is not supported")
    if vals_bdim is not None:
        raise NotImplementedError("Batching over matrix values is not supported yet")
    if b_bdim is None:
        return linear_solver_solve_p.bind(handle, pc_handle, shape, rows, cols, vals, b), None

    # Correctness-first batching for jacfwd/vmap. This launches one PETSc solve
    # per RHS; replace with a true multi-RHS/block solve when performance matters.
    b_batch = jnp.moveaxis(b, b_bdim, 0)
    x_batch = jnp.stack(
        [linear_solver_solve_p.bind(handle, pc_handle, shape, rows, cols, vals, b_batch[i]) for i in range(b_batch.shape[0])],
        axis=0,
    )
    return x_batch, 0


linear_solver_solve_p.def_impl(_linear_solver_solve_impl)
linear_solver_solve_p.def_abstract_eval(_linear_solver_solve_abstract_eval)
mlir.register_lowering(linear_solver_solve_p, _linear_solver_solve_lowering)

ad.primitive_jvps[linear_solver_solve_p] = _linear_solver_solve_jvp
ad.primitive_transposes[linear_solver_solve_p] = _linear_solver_solve_transpose
batching.primitive_batchers[linear_solver_solve_p] = _linear_solver_solve_batch
