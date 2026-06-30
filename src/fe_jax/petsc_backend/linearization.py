"""Callable-to-JAX-object adapters for Nathan/contact nonlinear solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.experimental.sparse as jsparse


@dataclass(frozen=True)
class Residual:
    """Residual callable wrapper matching fea-in-jax's linear_solve API."""

    function: Callable[[jax.Array], jax.Array]
    dirichlet_bcs_builtin: bool = False


@dataclass(frozen=True)
class Jacobian:
    """Jacobian callable wrapper matching fea-in-jax's linear_solve API."""

    function: Callable[[jax.Array], jsparse.COO]
    dirichlet_bcs_builtin: bool = False


@dataclass(frozen=True)
class JacobianDiagonal:
    """Jacobian diagonal callable wrapper matching fea-in-jax's linear_solve API."""

    function: Callable[[jax.Array], jax.Array]
    dirichlet_bcs_builtin: bool = False


# Compatibility with the spelling currently present in fea-in-jax.
JacobianDiagonl = JacobianDiagonal


@dataclass(frozen=True)
class LinearizationCallables:
    """Dirichlet-aware callable objects built from residual/Jacobian wrappers."""

    R_w_dirichlet: Callable[[jax.Array], jax.Array]
    J_w_dirichlet: Optional[Callable[[jax.Array], jsparse.COO]]
    diag_J_w_dirichlet: Optional[Callable[[jax.Array], jax.Array]]
    J_vp: Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class LinearSystemObjects:
    """Concrete JAX objects evaluated from nonlinear residual/Jacobian callables."""

    J_sparse: Optional[jsparse.COO]
    R_0: jax.Array
    rhs: jax.Array
    diag_J: Optional[jax.Array]
    J_vp: Callable[[jax.Array], jax.Array]


def as_residual(function: Callable[[jax.Array], jax.Array], *, dirichlet_bcs_builtin: bool = False):
    return Residual(function=function, dirichlet_bcs_builtin=dirichlet_bcs_builtin)


def as_jacobian(function: Callable[[jax.Array], jsparse.COO], *, dirichlet_bcs_builtin: bool = False):
    return Jacobian(function=function, dirichlet_bcs_builtin=dirichlet_bcs_builtin)


def as_jacobian_diagonal(
    function: Callable[[jax.Array], jax.Array],
    *,
    dirichlet_bcs_builtin: bool = False,
):
    return JacobianDiagonal(function=function, dirichlet_bcs_builtin=dirichlet_bcs_builtin)


def residual_to_callable(
    residual: Residual,
    constraints: Any,
    *args,
    apply_dirichlet_residual_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    **kwargs,
):
    """Return R(x) with boundary conditions applied."""
    if residual.dirichlet_bcs_builtin:
        return lambda x: residual.function(x, *args, **kwargs)

    if apply_dirichlet_residual_fn is not None:
        return lambda x: apply_dirichlet_residual_fn(residual.function(x, *args, **kwargs), x)

    if hasattr(constraints, "apply_to_residual"):
        return lambda x: constraints.apply_to_residual(residual.function(x, *args, **kwargs), x)

    raise ValueError("Residual needs Dirichlet handling, but no residual helper was provided.")


def jacobian_to_callable(
    jacobian: Optional[Jacobian],
    constraints: Any,
    *args,
    apply_dirichlet_lhs_fn: Optional[Callable[[jsparse.COO, jax.Array], jsparse.COO]] = None,
    **kwargs,
):
    """Return J(x) with boundary conditions applied, or None for matrix-free use."""
    if jacobian is None:
        return None

    if jacobian.dirichlet_bcs_builtin:
        return lambda x: jacobian.function(x, *args, **kwargs)

    if apply_dirichlet_lhs_fn is None:
        raise ValueError("Jacobian needs Dirichlet handling, but no LHS helper was provided.")

    return lambda x: apply_dirichlet_lhs_fn(
        jacobian.function(x, *args, **kwargs),
        constraints.dep_dofs,
    )


def jacobian_diagonal_to_callable(
    jacobian_diagonal: Optional[JacobianDiagonal],
    constraints: Any,
    *args,
    **kwargs,
):
    """Return diag(J)(x) with constrained entries forced to one."""
    if jacobian_diagonal is None:
        return None

    if jacobian_diagonal.dirichlet_bcs_builtin:
        return lambda x: jacobian_diagonal.function(x, *args, **kwargs)

    return lambda x: jacobian_diagonal.function(x, *args, **kwargs).at[constraints.dep_dofs].set(1.0)


def jvp_callable_from_residual(R_w_dirichlet: Callable[[jax.Array], jax.Array], x_0: jax.Array):
    """Bind x_0 into a Jacobian-vector-product callable."""
    return jax.tree_util.Partial(
        lambda x, z: jax.jvp(R_w_dirichlet, (x,), (z,))[1],
        x_0,
    )


def evaluate_linearization_callables(
    linearization: LinearizationCallables,
    x_0: jax.Array,
) -> LinearSystemObjects:
    """Evaluate callable residual/Jacobian objects at x_0."""
    R_0 = linearization.R_w_dirichlet(x_0)
    J_sparse = None
    if linearization.J_w_dirichlet is not None:
        J_sparse = linearization.J_w_dirichlet(x_0)

    diag_J = None
    if linearization.diag_J_w_dirichlet is not None:
        diag_J = linearization.diag_J_w_dirichlet(x_0)

    return LinearSystemObjects(
        J_sparse=J_sparse,
        R_0=R_0,
        rhs=-R_0,
        diag_J=diag_J,
        J_vp=linearization.J_vp,
    )


def callables_to_linear_system_objects(
    *,
    R_w_dirichlet: Callable[[jax.Array], jax.Array],
    J_w_dirichlet: Optional[Callable[[jax.Array], jsparse.COO]],
    x_0: jax.Array,
    diag_J_w_dirichlet: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> LinearSystemObjects:
    """Convert already-prepared Nathan callables into concrete JAX objects."""
    linearization = LinearizationCallables(
        R_w_dirichlet=R_w_dirichlet,
        J_w_dirichlet=J_w_dirichlet,
        diag_J_w_dirichlet=diag_J_w_dirichlet,
        J_vp=jvp_callable_from_residual(R_w_dirichlet, x_0),
    )
    return evaluate_linearization_callables(linearization, x_0)


def build_linearization_callables(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonal],
    constraints: Any,
    x_0: jax.Array,
    *args,
    apply_dirichlet_lhs_fn: Optional[Callable[[jsparse.COO, jax.Array], jsparse.COO]] = None,
    apply_dirichlet_residual_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    **kwargs,
) -> LinearizationCallables:
    """Build the callable objects that Nathan's pipeline creates inside linear_solve."""
    R_w_dirichlet = residual_to_callable(
        residual,
        constraints,
        *args,
        apply_dirichlet_residual_fn=apply_dirichlet_residual_fn,
        **kwargs,
    )
    J_w_dirichlet = jacobian_to_callable(
        jacobian,
        constraints,
        *args,
        apply_dirichlet_lhs_fn=apply_dirichlet_lhs_fn,
        **kwargs,
    )
    diag_J_w_dirichlet = jacobian_diagonal_to_callable(
        jacobian_diagonal,
        constraints,
        *args,
        **kwargs,
    )

    return LinearizationCallables(
        R_w_dirichlet=R_w_dirichlet,
        J_w_dirichlet=J_w_dirichlet,
        diag_J_w_dirichlet=diag_J_w_dirichlet,
        J_vp=jvp_callable_from_residual(R_w_dirichlet, x_0),
    )


def build_linear_system_objects(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonal],
    constraints: Any,
    x_0: jax.Array,
    *args,
    apply_dirichlet_lhs_fn: Optional[Callable[[jsparse.COO, jax.Array], jsparse.COO]] = None,
    apply_dirichlet_residual_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    **kwargs,
) -> LinearSystemObjects:
    """Build and immediately evaluate Nathan-style callables into JAX objects."""
    linearization = build_linearization_callables(
        residual,
        jacobian,
        jacobian_diagonal,
        constraints,
        x_0,
        *args,
        apply_dirichlet_lhs_fn=apply_dirichlet_lhs_fn,
        apply_dirichlet_residual_fn=apply_dirichlet_residual_fn,
        **kwargs,
    )
    return evaluate_linearization_callables(linearization, x_0)
