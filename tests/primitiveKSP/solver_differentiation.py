import jax
import jax.numpy as jnp
from jax.interpreters import ad, batching

from .solver_creation import CupyKSPCtx, petsc_cleanup, petsc_init
from .solver_usage import petsc_solve, petsc_solve_handle, petsc_solve_transpose_handle


try:
    from jax.extend import core as jax_core
except ImportError:
    from jax import core as jax_core

try:
    from jax import core as legacy_core
except ImportError:
    legacy_core = jax_core


linear_solve_p = jax_core.Primitive("petsc_linear_solve")
solve_from_coo_p = jax_core.Primitive("petsc_solve_from_coo")


def _linear_solve_impl(handle, b):
    return petsc_solve_handle(handle, b)


def _linear_solve_abstract_eval(handle_aval, b_aval):
    del handle_aval
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array(b_aval.shape, b_aval.dtype)


linear_solve_p.def_impl(_linear_solve_impl)
linear_solve_p.def_abstract_eval(_linear_solve_abstract_eval)


def linear_solve(solver: CupyKSPCtx, b: jnp.ndarray):
    return linear_solve_p.bind(solver.handle, b)


def solve_from_coo(shape, rows, cols, vals, b):
    return solve_from_coo_p.bind(shape, rows, cols, vals, b)


def _zero_from_value(value):
    if hasattr(ad.Zero, "from_primal_value"):
        return ad.Zero.from_primal_value(value)
    return ad.Zero(legacy_core.get_aval(value))


def _is_zero(value):
    return type(value) is ad.Zero


def _coo_matvec(rows, cols, vals, x):
    return jnp.zeros_like(x).at[rows].add(vals * x[cols])


def _solve_from_coo_impl(shape, rows, cols, vals, b):
    # Lifecycle-owning primitive: PETSc setup/teardown is implementation detail,
    # while derivative rules below expose only the math of x = A^{-1} b to JAX.
    solver = petsc_init(shape, vals, rows, cols)
    try:
        return petsc_solve(solver, b)
    finally:
        petsc_cleanup(solver)


def _solve_from_coo_abstract_eval(shape_aval, rows_aval, cols_aval, vals_aval, b_aval):
    del shape_aval, rows_aval, cols_aval, vals_aval
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array(b_aval.shape, b_aval.dtype)


solve_from_coo_p.def_impl(_solve_from_coo_impl)
solve_from_coo_p.def_abstract_eval(_solve_from_coo_abstract_eval)


def _linear_solve_jvp(primals, tangents):
    handle, b = primals
    _, b_dot = tangents

    x = linear_solve_p.bind(handle, b)
    if _is_zero(b_dot):
        x_dot = _zero_from_value(x)
    else:
        x_dot = linear_solve_p.bind(handle, b_dot)

    return x, x_dot


def _linear_solve_transpose(ct, handle, b):
    handle_bar = None

    if _is_zero(ct):
        b_bar = None if not isinstance(b, ad.UndefinedPrimal) else ad.Zero(b.aval)
    else:
        b_bar = petsc_solve_transpose_handle(handle, ct)

    return handle_bar, b_bar


def _solve_from_coo_jvp(primals, tangents):
    shape, rows, cols, vals, b = primals
    _, _, _, vals_dot, b_dot = tangents

    x = solve_from_coo_p.bind(shape, rows, cols, vals, b)

    if _is_zero(vals_dot) and _is_zero(b_dot):
        return x, _zero_from_value(x)

    rhs_dot = jnp.zeros_like(b) if _is_zero(b_dot) else b_dot
    if not _is_zero(vals_dot):
        rhs_dot = rhs_dot - _coo_matvec(rows, cols, vals_dot, x)

    x_dot = solve_from_coo_p.bind(shape, rows, cols, vals, rhs_dot)
    return x, x_dot


def _solve_from_coo_transpose(ct, shape, rows, cols, vals, b):
    shape_bar = None
    rows_bar = None
    cols_bar = None

    vals_is_unknown = isinstance(vals, ad.UndefinedPrimal)
    b_is_unknown = isinstance(b, ad.UndefinedPrimal)

    if _is_zero(ct):
        vals_bar = ad.Zero(vals.aval) if vals_is_unknown else None
        b_bar = ad.Zero(b.aval) if b_is_unknown else None
        return shape_bar, rows_bar, cols_bar, vals_bar, b_bar

    if vals_is_unknown:
        raise NotImplementedError(
            "petsc_solve_from_coo transpose needs primal vals to build the PETSc transpose solve"
        )

    solver = petsc_init(shape, vals, rows, cols)
    try:
        adjoint = petsc_solve_transpose_handle(solver.handle, ct)
        if b_is_unknown:
            vals_bar = None
        else:
            x = petsc_solve(solver, b)
            vals_bar = -adjoint[rows] * x[cols]
    finally:
        petsc_cleanup(solver)

    b_bar = adjoint if b_is_unknown else None
    return shape_bar, rows_bar, cols_bar, vals_bar, b_bar


def _linear_solve_batch(args, batch_dims):
    handle, b = args
    handle_bdim, b_bdim = batch_dims

    if handle_bdim is not None:
        raise NotImplementedError("Batching over PETSc solver handles is not supported")

    if b_bdim is None:
        return linear_solve_p.bind(handle, b), None

    # Correctness-first batching for jacfwd/vmap. This launches one PETSc solve
    # per RHS; replace with a true multi-RHS/block solve when performance matters.
    b_batch = jnp.moveaxis(b, b_bdim, 0)
    x_batch = jnp.stack(
        [linear_solve_p.bind(handle, b_batch[i]) for i in range(b_batch.shape[0])],
        axis=0,
    )
    return x_batch, 0


def _solve_from_coo_batch(args, batch_dims):
    shape, rows, cols, vals, b = args
    shape_bdim, rows_bdim, cols_bdim, vals_bdim, b_bdim = batch_dims

    if shape_bdim is not None or rows_bdim is not None or cols_bdim is not None:
        raise NotImplementedError("Batching over COO metadata is not supported")
    if vals_bdim is not None:
        raise NotImplementedError("Batching over matrix values is not supported yet")
    if b_bdim is None:
        return solve_from_coo_p.bind(shape, rows, cols, vals, b), None

    # Correctness-first batching for jacfwd/vmap. This launches one PETSc solve
    # per RHS; replace with a true multi-RHS/block solve when performance matters.
    b_batch = jnp.moveaxis(b, b_bdim, 0)
    x_batch = jnp.stack(
        [solve_from_coo_p.bind(shape, rows, cols, vals, b_batch[i]) for i in range(b_batch.shape[0])],
        axis=0,
    )
    return x_batch, 0


ad.primitive_jvps[linear_solve_p] = _linear_solve_jvp
ad.primitive_transposes[linear_solve_p] = _linear_solve_transpose
batching.primitive_batchers[linear_solve_p] = _linear_solve_batch

ad.primitive_jvps[solve_from_coo_p] = _solve_from_coo_jvp
ad.primitive_transposes[solve_from_coo_p] = _solve_from_coo_transpose
batching.primitive_batchers[solve_from_coo_p] = _solve_from_coo_batch
