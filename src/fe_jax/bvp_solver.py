from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import jax.numpy as jnp
import numpy as np

import jetsci
from jetsci import petsc_snes

from .boundary_conditions import DirichletBC, NeumannBC, PeriodicBC
from .constraints import MultiPointConstraint
from .element_batch import ElementBatch
from .fea import build_differentiable_bvp_PETSc_solve, solve_bvp, solve_bvp_PETSc
from .sparse_linear_solve import SolverOptions as JAXSolverOptions


class BVPBackend(Enum):
    JAX = auto()
    PETSC = auto()


@dataclass
class UnifiedBVPOptions:
    backend: BVPBackend = BVPBackend.JAX
    jax_solver_options: JAXSolverOptions | None = None
    petsc_solver_options: jetsci.SolverOptions | None = None
    diagnostics: bool = False


def _flatten_material_params(element_batches: list[ElementBatch]) -> jnp.ndarray:
    return jnp.hstack([batch.material_params.ravel() for batch in element_batches])


def _replace_material_params(
    element_batches: list[ElementBatch],
    material_params: jnp.ndarray,
) -> list[ElementBatch]:
    material_params = jnp.asarray(material_params)
    offset = 0
    updated_batches = []
    for batch in element_batches:
        size = batch.material_params.size
        batch_params = material_params[offset : offset + size].reshape(
            batch.material_params.shape
        )
        updated_batches.append(batch.replace(material_params=batch_params))
        offset += size

    if offset != material_params.size:
        raise ValueError(
            "Material parameter vector has size "
            f"{material_params.size}, but expected {offset}."
        )
    return updated_batches


def _default_x0(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    global_values: list[int] | None,
) -> jnp.ndarray:
    if vertices_vd.ndim == 1:
        n_vertices = vertices_vd.shape[0]
    else:
        n_vertices = vertices_vd.shape[0]
    n_solution_components = element_batches[0].n_dofs_per_basis
    n_global_values = sum(global_values or [])
    return jnp.zeros((n_vertices * n_solution_components + n_global_values,))


@dataclass
class UnifiedBVPSolver:
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]]
    element_batches: list[ElementBatch]
    element_residual_func: Any
    boundary_conditions: list[DirichletBC | NeumannBC | PeriodicBC] | None = None
    multipoint_constraints: list[MultiPointConstraint] | None = None
    global_values: list[int] | None = None
    u_0_g: jnp.ndarray | None = None
    options: UnifiedBVPOptions = field(default_factory=UnifiedBVPOptions)

    def __post_init__(self):
        if self.options.jax_solver_options is None:
            self.options.jax_solver_options = JAXSolverOptions()

    @property
    def phi(self) -> jnp.ndarray:
        return _flatten_material_params(self.element_batches)

    @property
    def x0(self) -> jnp.ndarray:
        if self.u_0_g is not None:
            return self.u_0_g
        return _default_x0(self.vertices_vd, self.element_batches, self.global_values)

    def solve(
        self,
        *,
        material_params: jnp.ndarray | None = None,
        u_0_g: jnp.ndarray | None = None,
    ):
        element_batches = self.element_batches
        if material_params is not None:
            element_batches = _replace_material_params(element_batches, material_params)
        x0 = self.x0 if u_0_g is None else u_0_g

        match self.options.backend:
            case BVPBackend.JAX:
                return solve_bvp(
                    vertices_vd=self.vertices_vd,
                    element_batches=element_batches,
                    element_residual_func=self.element_residual_func,
                    boundary_conditions=self.boundary_conditions,
                    multipoint_constraints=self.multipoint_constraints,
                    global_values=self.global_values,
                    u_0_g=x0,
                    solver_options=self.options.jax_solver_options,
                )
            case BVPBackend.PETSC:
                result = solve_bvp_PETSc(
                    vertices_vd=self.vertices_vd,
                    element_batches=element_batches,
                    element_residual_func=self.element_residual_func,
                    boundary_conditions=self.boundary_conditions,
                    multipoint_constraints=self.multipoint_constraints,
                    global_values=self.global_values,
                    u_0_g=x0,
                    diagnostics=self.options.diagnostics,
                    petsc_solver_options=self.options.petsc_solver_options,
                    destroy_solver=False,
                    return_petsc_solver_options=True,
                )
                self.options.petsc_solver_options = result[3]
                return result[:3]
            case _:
                raise ValueError(f"Unsupported BVP backend {self.options.backend!r}")

    def build_material_parameter_solve(self):
        """Return `(solve_phi, phi0, x0)` with backend-specific lifecycle attached."""
        match self.options.backend:
            case BVPBackend.JAX:

                def solve_phi(phi, x0):
                    u, _, _ = self.solve(material_params=phi, u_0_g=x0)
                    return u

                return solve_phi, self.phi, self.x0

            case BVPBackend.PETSC:
                solve_phi, phi0, x0, _, petsc_options = build_differentiable_bvp_PETSc_solve(
                    vertices_vd=self.vertices_vd,
                    element_batches=self.element_batches,
                    element_residual_func=self.element_residual_func,
                    boundary_conditions=self.boundary_conditions,
                    multipoint_constraints=self.multipoint_constraints,
                    global_values=self.global_values,
                    u_0_g=self.x0,
                    diagnostics=self.options.diagnostics,
                    petsc_solver_options=self.options.petsc_solver_options,
                )
                self.options.petsc_solver_options = petsc_options
                return solve_phi, phi0, x0
            case _:
                raise ValueError(f"Unsupported BVP backend {self.options.backend!r}")

    def destroy(self):
        solver_key = None
        if self.options.petsc_solver_options is not None:
            solver_key = self.options.petsc_solver_options.solver_key
        if solver_key is not None:
            petsc_snes.differentiable_snes.unregister_primitive_context(solver_key)
            petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)
            self.options.petsc_solver_options = None


def build_bvp_solver(
    *,
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: Any,
    boundary_conditions: list[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: list[MultiPointConstraint] | None = None,
    global_values: list[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    options: UnifiedBVPOptions | None = None,
) -> UnifiedBVPSolver:
    return UnifiedBVPSolver(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
        u_0_g=u_0_g,
        options=options or UnifiedBVPOptions(),
    )
