"""Autodiff boundary for PETSc SNES-style nonlinear solves.

This module is the place where we keep the mathematical differentiation rule
for a nonlinear solve, separate from the PETSc object lifecycle and callback
plumbing.

For a residual

    R(phi, x_star(phi)) = 0

the implicit-function theorem gives

    R_x x_dot = -R_phi phi_dot.

That is the rule implemented here. The primal nonlinear solve is supplied as a
hook so the same differentiation boundary can be backed by PETSc SNES, a pure
JAX test solve, or another solve engine while we sort out the plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

import cupy as cp

import sys
from time import perf_counter

from jax.experimental.buffer_callback import buffer_callback

from petsc4py import PETSc

from .solver_lifecycle import *

try:
    from jax.extend import core as jax_core
except ImportError:  # pragma: no cover - older JAX fallback
    from jax import core as jax_core


ResidualFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
NonlinearSolve = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
LinearSolve = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
TransposeLinearSolve = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]

#Figure out  a more elegant way to do this, because having another dictionary is kinda boof
_PRIMITIVE_CONTEXTS: dict[int, "DifferentiableSNESPrimitive"] = {}
_JVP_DIAGNOSTICS = False


def set_jvp_diagnostics(enabled: bool = True) -> None:
    """Enable timing output from the primitive JVP rule."""
    global _JVP_DIAGNOSTICS
    _JVP_DIAGNOSTICS = enabled


def _block_if_ready(value):
    block_until_ready = getattr(value, "block_until_ready", None)
    if block_until_ready is not None:
        block_until_ready()

try:
    _SHAPED_ARRAY = jax_core.ShapedArray
except AttributeError:  # pragma: no cover - older JAX fallback
    from jax import core as legacy_core

    _SHAPED_ARRAY = legacy_core.ShapedArray


_differentiable_snes_solve_p = jax_core.Primitive("jetsci_differentiable_snes_solve")


def _is_zero(value):
    return type(value) is ad.Zero


def _zero_from_value(value):
    if hasattr(ad.Zero, "from_primal_value"):
        return ad.Zero.from_primal_value(value)
    return ad.Zero(_SHAPED_ARRAY(value.shape, value.dtype))


def _primitive_context(solver_key: int) -> "DifferentiableSNESPrimitive":
    try:
        return _PRIMITIVE_CONTEXTS[solver_key]
    except KeyError as exc:
        raise KeyError(
            f"No differentiable SNES primitive context registered for solver_key={solver_key}"
        ) from exc


@dataclass(frozen=True)
class DifferentiableSNESPrimitive:
    """Callables used by the differentiable nonlinear-solve boundary.

    Parameters
    ----------
    residual:
        Function of `(phi, x)` returning the nonlinear residual.
    nonlinear_solve:
        Primal solve function of `(phi, x0)` returning the converged state.
        This may be omitted when `solver_key` is provided and the live PETSc
        solver hooks should be fetched from the solver dictionary instead.
    linear_solve:
        Optional linear solve hook used for the forward-mode IFT solve.
    transpose_linear_solve:
        Optional linear solve hook for the transpose system needed by reverse
        mode. This is carried here now so the file is ready when we wire VJP in.
    """

    residual: ResidualFunction
    nonlinear_solve: NonlinearSolve | None = None
    linear_solve: LinearSolve | None = None
    transpose_linear_solve: TransposeLinearSolve | None = None
    solver_key: int | None = None
    # Optional phi-aware Jacobian, retained at the end for positional API
    # compatibility with the earlier primitive constructor.
    jacobian: Callable | None = None




def _hooks_from_solver_key(
    solver_key: int,
    primitive: "DifferentiableSNESPrimitive",
    linear_solve: LinearSolve | None = None,
    transpose_linear_solve: TransposeLinearSolve | None = None,
):
    """Build solve hooks from a live solver dictionary entry."""
    snes_solver, ksp_solver = get_petsc_solver_objects_from_key(solver_key)

    def nonlinear_solve(phi, x0):
        # The PETSc callbacks are closures over the active problem data.  A
        # solver-key primitive may be called with a different phi on every
        # invocation, so refresh those closures while retaining the SNES/KSP
        # objects and their allocated storage.
        residual_for_phi = jax.tree_util.Partial(primitive.residual, phi)
        if primitive.jacobian is not None:
            jacobian_for_phi = jax.tree_util.Partial(primitive.jacobian, phi)
        else:
            jacobian_for_phi = jax.tree_util.Partial(
                lambda active_x: jax.jacfwd(
                    lambda x: primitive.residual(phi, x) #fix these so their not lambdass
                )(active_x)
            )
        refresh_start = perf_counter()
        update_petsc_snes_callbacks(
            snes_solver,
            residual_for_phi,
            jacobian_for_phi,
        )
        refresh_time = perf_counter() - refresh_start
        if _JVP_DIAGNOSTICS:
            print("JVP SNES callback refresh:", refresh_time)
        return snes_solver.solve_to_jax(x0)

    
    #okay for now this version works for sequential vector calls (one per column of the Jacobian)
    def _linear_solve_callback(ctx, out, rhs):

        print("inside _linear_solve_callback")

        GPUPointerArray = cp.from_dlpack(rhs,copy=False)
        #print("cupy array",GPUPointerArray)

        #print("working shape",GPUPointerArray.ndim)

        #print("GPUPointerArray",GPUPointerArray)

        """
        rhs_block = PETSc.Mat().createDenseCUDA(GPUPointerArray.shape[0])

        rhs_block.setPreallocationDense(jnp.zeros((3,3)))
        
        


        rhs_block.view()

        """

        

        if(GPUPointerArray.ndim == 1):
            #rhs_petsc = PETSc.Vec().createWithDLPack(GPUPointerArray, size=rhs.shape[0]) #we ought to make this swap otherwise we gain nothing

            x_gpu = ksp_solver.solve_to_jax(rhs) #this is lazy and inefficient
        else:
            rhs_block = PETSc.Mat().createDenseCUDA(GPUPointerArray.shape)
            rhs_block.setPreallocationDense(jnp.asarray(rhs))
                    
            print("built block RHS")

            x_petsc = ksp_solver.block_linear_solve(rhs_block) #this is lazy and inefficient

            """cudahandle = x_petsc.getCUDAHandle()
            ptr = cudahandle         # raw CUDA pointer from PETSc
            length = x_petsc.getSize()
            
            x_gpu = cp.ndarray((length,), dtype=cp.float64 , memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, length*8, x_petsc), 0))
            """

            x_gpu = jnp.array(x_petsc.getDenseArray())


        cp.asarray(out)[...] = x_gpu

    def _linear_solve_callback_block(ctx, out, rhs_block: jnp.ndarray):

        GPUPointerArray = cp.from_dlpack(rhs_block,copy=False)

        rhs_petsc = PETSc.Vec().createWithDLPack(GPUPointerArray, size=rhs_block.shape[0])
        
        #rhs_petsc.view()

        x_gpu = ksp_solver.solve_to_jax(rhs_block)

        cp.asarray(out)[...] = x_gpu


    def _transpose_linear_solve_callback(rhs):
        # TODO: remove the NumPy bridge once the solver-key linear solve path
        # can stay device-native end to end.
        return np.asarray(ksp_solver.solve_transpose_to_jax(rhs))

    
    def _ksp_linear_solve_buffer_callback(rhs):
        rhs = jnp.asarray(rhs)
        print("_ksp_linear_solve_single_rhs",rhs.shape)
        result_shape = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        
        return buffer_callback(
            _linear_solve_callback,
            result_shape,
            vmap_method="expand_dims")(rhs)
    

    def _ksp_transpose_linear_solve_single_rhs(rhs):
        rhs = jnp.asarray(rhs)
        result_shape = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        return buffer_callback(
            _transpose_linear_solve_callback,
            result_shape,
            rhs,
        )

    
    def ksp_linear_solve(x_star, phi, rhs):
        del x_star, phi
        print("inside ksp_linear_solve")
        if snes_solver.jacobian_mat is not None:
            ksp_solver.update_operator(snes_solver.jacobian_mat)
        rhs = jnp.asarray(rhs)
        return _ksp_linear_solve_buffer_callback(rhs)
        #return jax.vmap(_ksp_linear_solve_single_rhs, in_axes=0, out_axes=0)(rhs)
    
        
    def ksp_transpose_linear_solve(x_star, phi, rhs):
        del x_star, phi
        if snes_solver.jacobian_mat is not None:
            ksp_solver.update_operator(snes_solver.jacobian_mat)
        rhs = jnp.asarray(rhs)
        if rhs.ndim == 1:
            return _ksp_transpose_linear_solve_single_rhs(rhs)
        #return jax.vmap(_ksp_transpose_linear_solve_single_rhs, in_axes=0, out_axes=0)(rhs)


    #we're going to convert this to a block call in a minute

    def ksp_linear_solve_block(x_star, phi, rhs):
        jax.debug.print("called ksp_linear_solve")

        rhs = jnp.asarray(rhs)
        jax.debug.print("rhs ndim {rhsdim}",rhsdim=rhs.ndim)
        jax.debug.print("rhs shape {rhsshape}",rhsshape=rhs.shape)
        jax.debug.print("rhs type {rhstype}",rhstype=type(rhs))
        exit(1)


    return (
        nonlinear_solve,
        linear_solve or ksp_linear_solve,
        transpose_linear_solve or ksp_transpose_linear_solve,
    )


def _register_primitive_context(primitive: "DifferentiableSNESPrimitive") -> None:
    if primitive.solver_key is None:
        raise ValueError("solver_key is required to register a differentiable SNES primitive context")
    _PRIMITIVE_CONTEXTS[primitive.solver_key] = primitive


def unregister_primitive_context(solver_key: int) -> None:
    """Remove a registered primitive context when its PETSc solver is destroyed."""
    _PRIMITIVE_CONTEXTS.pop(solver_key, None)


def _differentiable_snes_solve_impl(phi, x0, *, solver_key: int):
    primitive = _primitive_context(solver_key)
    if primitive.solver_key is not None and primitive.solver_key != solver_key:
        raise KeyError(
            f"Primitive registry mismatch for solver_key={solver_key}: "
            f"registered context has solver_key={primitive.solver_key}"
        )
    nonlinear_solve, _, _ = _hooks_from_solver_key(
        solver_key,
        primitive,
        primitive.linear_solve,
        primitive.transpose_linear_solve,
    )
    return nonlinear_solve(phi, jax.lax.stop_gradient(x0))


def _differentiable_snes_solve_abstract_eval(phi_aval, x0_aval, *, solver_key: int):
    del phi_aval, solver_key
    return _SHAPED_ARRAY(x0_aval.shape, x0_aval.dtype)


def _differentiable_snes_solve_jvp(primals, tangents, *, solver_key: int):
    total_start = perf_counter()
    phi, x0 = primals
    phi_dot, _x0_dot = tangents

    jax.debug.print("we're in here actually")

    primitive = _primitive_context(solver_key)
    hook_start = perf_counter()
    nonlinear_solve, linear_solve, _ = _hooks_from_solver_key(
        solver_key,
        primitive,
        primitive.linear_solve,
        primitive.transpose_linear_solve,
    )
    hook_time = perf_counter() - hook_start

    primal_start = perf_counter()
    x_star = nonlinear_solve(phi, jax.lax.stop_gradient(x0))
    _block_if_ready(x_star)
    primal_time = perf_counter() - primal_start

    if _is_zero(phi_dot):
        return x_star, _zero_from_value(x_star)

    residual_at_solution = jax.tree_util.Partial(primitive.residual, x=x_star)

    residual_start = perf_counter()         #this is taking an absurdly long time
    _, residual_phi_dot = jax.jvp(          #figure out why it's taking so long when
        residual_at_solution,               #presumably JAX is doing the same thing in GMRES/NEWTON
        (phi,),
        (phi_dot,),
    )
    _block_if_ready(residual_phi_dot)
    residual_time = perf_counter() - residual_start

    linear_start = perf_counter()
    x_dot = linear_solve(x_star, phi, -residual_phi_dot)
    _block_if_ready(x_dot)
    linear_time = perf_counter() - linear_start

    if _JVP_DIAGNOSTICS:
        print(
            "JVP diagnostics:",
            {
                "hook_setup_s": hook_time,
                "primal_snes_s": primal_time,
                "residual_jvp_s": residual_time,
                "companion_ksp_s": linear_time,
                "total_s": perf_counter() - total_start,
                "rhs_shape": getattr(residual_phi_dot, "shape", None),
            },
        )
    return x_star, x_dot


def _differentiable_snes_solve_transpose(ct, phi, x0, *, solver_key: int):
    primitive = _primitive_context(solver_key)
    nonlinear_solve, _, transpose_linear_solve = _hooks_from_solver_key(
        solver_key,
        primitive,
        primitive.linear_solve,
        primitive.transpose_linear_solve,
    )

    if _is_zero(ct):
        phi_bar = _zero_from_value(phi)
        x0_bar = _zero_from_value(x0)
        return phi_bar, x0_bar

    x_star = nonlinear_solve(phi, jax.lax.stop_gradient(x0))
    lambda_vec = transpose_linear_solve(x_star, phi, ct)

    def residual_at_phi(active_phi):
        return primitive.residual(active_phi, x_star)

    _, residual_vjp = jax.vjp(residual_at_phi, phi)
    phi_bar = residual_vjp(-lambda_vec)[0]
    x0_bar = _zero_from_value(x0)
    return phi_bar, x0_bar


_differentiable_snes_solve_p.def_impl(_differentiable_snes_solve_impl)
_differentiable_snes_solve_p.def_abstract_eval(_differentiable_snes_solve_abstract_eval)
ad.primitive_jvps[_differentiable_snes_solve_p] = _differentiable_snes_solve_jvp
ad.primitive_transposes[_differentiable_snes_solve_p] = _differentiable_snes_solve_transpose


def make_differentiable_snes_solve(primitive: DifferentiableSNESPrimitive):
    """Return `solve(phi, x0)` with forward and reverse IFT rules.

    The solver-key path uses a registered JAX primitive so both jacfwd and
    jacrev can be wired through PETSc KSP solves. The solver-free fallback keeps
    the earlier dense correctness-first custom JVP path for smoke tests.
    """

    residual = primitive.residual


    if primitive.solver_key is not None:
        _register_primitive_context(primitive)
        solver_key = primitive.solver_key

        def solve(phi, x0):
            return _differentiable_snes_solve_p.bind(phi, x0, solver_key=solver_key)

        return solve

    if primitive.nonlinear_solve is None:
        raise ValueError(
            "DifferentiableSNESPrimitive.nonlinear_solve is required when solver_key is not set"
        )

    nonlinear_solve = primitive.nonlinear_solve
    linear_solve = primitive.linear_solve #or _dense_linear_solve_from_residual(residual)
    transpose_linear_solve = (
        primitive.transpose_linear_solve
    )

    
    def solve(phi, x0):
        
        x_star = nonlinear_solve(phi, jax.lax.stop_gradient(x0))
        return jax.lax.stop_gradient(x_star)

    #This is an old version, I don't think we need it anymore but we'll hold on to it for now
    """
    @solve.defjvp
    def solve_jvp(primals, tangents):
        phi, x0 = primals
        phi_dot, _x0_dot = tangents

        jax.debug.print("inside solve_jvp")
        x_star = solve(phi, x0)

        if type(phi_dot).__name__ == "Zero":
            return x_star, jnp.zeros_like(x_star)

        def residual_at_solution(active_phi):
            return residual(active_phi, x_star)

        _, residual_phi_dot = jax.jvp(
            residual_at_solution,
            (phi,),
            (phi_dot,),
        )
        x_dot = linear_solve(x_star, phi, -residual_phi_dot)
        return x_star, x_dot
    """
    return solve


__all__ = [
    "DifferentiableSNESPrimitive",
    "LinearSolve",
    "NonlinearSolve",
    "ResidualFunction",
    "TransposeLinearSolve",
    "make_differentiable_snes_solve",
    "set_jvp_diagnostics",
]
