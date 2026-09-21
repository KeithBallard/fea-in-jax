"""Unit tests ensuring BVP solving and differentiable sensitivity analysis in fea-in-jax.

This module validates that `solve_bvp`, `build_differentiable_bvp_solve`, and
`UnifiedBVPSolver` are fully differentiable with respect to constitutive model
parameters using JetSCI's differentiable solver functionality. Both linear
elasticity and nonlinear hyperelasticity cases are tested and verified against
analytical solutions and finite difference approximations.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fe_jax import (
    solve_bvp,
    build_differentiable_bvp_solve,
    UnifiedBVPSolver,
    UnifiedBVPOptions,
    ElementBatch,
    FiniteElementType,
    CellType,
    ElementFamily,
    LagrangeVariant,
    QuadratureType,
    DirichletBC,
    NeumannBC,
    BCType,
    SolverOptions,
    NonlinearSolverType,
    LinearSolverType,
    linear_truss_residual,
    elastic_truss,
)
from fe_jax.hyperelasticity import hyperelasticity_residual, st_venant_kirchhoff


def test_linear_elasticity_differentiable_solve_bvp():
    """Verify solve_bvp is differentiable w.r.t constitutive parameters for linear elasticity.

    BVP Problem Description:
    ------------------------
    - Physical Problem: 1D linear elastic rod/bar under uniaxial tension.
      The bar has length L = 1.0, cross-sectional area A = 1.0, and Young's modulus E = 100.0.
    - Governing Differential Equation:
        d/dx [ E * A * du/dx ] = 0,   for x in (0, L)
    - Boundary Conditions:
        * Left end (x = 0, node 0): Fixed Dirichlet BC, u(0) = 0.0.
        * Right end (x = L, node 2): Axial Neumann tensile point load, F = 10.0.
    - FE Discretization:
        * Domain [0, 1] discretized into 2 linear (P1 Lagrange) interval elements with 3 nodes.
        * Quadrature: 2-point Gauss quadrature rule per element.
    - Closed-Form Analytical Solution:
        * Displacement field: u(x) = (F / (E * A)) * x
        * At node 0 (x = 0.0): u = 0.0
        * At node 1 (x = 0.5): u = 0.05
        * At node 2 (x = 1.0): u = 0.10
    - Analytical Parameter Sensitivities:
        * du/dE = -(F * x) / (E^2 * A) = -u(x) / E
        * du/dA = -(F * x) / (E * A^2) = -u(x) / A

    Targeted Features:
    ------------------
    - JetSCI Linear Solver Delegation: Tests that `solve_bvp` delegates the linear BVP solve
      to JetSCI's `differentiable_solve` utilizing the JAX CG linear solver (`CG_JAX_SCIPY_W_INFO`).
    - Forward-Mode Automatic Differentiation: Tests that `jax.jacobian` computes the exact
      sensitivity of the displacement solution vector with respect to Young's modulus E:
          du/dE = - [K]^{-1} [dK/dE * u]
      via the Implicit Function Theorem (IFT) tracked through `jetsci.differentiable_solve`.
    - Accuracy Verification: Compares the autodiff derivative against both the closed-form
      analytical derivative and central finite difference approximations.
    """
    points = np.linspace(0.0, 1.0, 3, dtype=np.float64).reshape((-1, 1))
    cells = np.array([[0, 1], [1, 2]], dtype=np.uint64)

    F_load = 10.0
    bcs = [
        DirichletBC(bc_type=BCType.NODE, component=0, index=0, value=0.0),
        NeumannBC(bc_type=BCType.NODE, component=0, index=2, value=F_load),
    ]

    fe_type = FiniteElementType(
        cell_type=CellType.interval,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    base_batch = ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=1,
        connectivity_en=cells,
        constitutive_model=elastic_truss,
        material_params=jnp.array([100.0, 1.0]),
    )

    def solve_linear(E_val):
        params = jnp.array([E_val, 1.0])
        batches = [base_batch.replace(material_params=params)]
        u, _, _ = solve_bvp(
            vertices_vd=points,
            element_batches=batches,
            element_residual_func=linear_truss_residual,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            ),
        )
        return u

    E0 = 100.0
    u0 = solve_linear(E0)

    # 1. Analytical check for primal solution
    expected_u = jnp.array([0.0, 0.05, 0.1])
    assert jnp.allclose(u0, expected_u, atol=1e-6)

    # 2. Autodiff derivative computation via Implicit Function Theorem in solve_bvp
    du_dE = jax.jacobian(solve_linear)(E0)
    expected_du_dE = -u0 / E0
    assert jnp.allclose(du_dE, expected_du_dE, rtol=1e-4, atol=1e-6)

    # 3. Finite difference verification
    eps = 1e-4
    u_plus = solve_linear(E0 + eps)
    u_minus = solve_linear(E0 - eps)
    du_dE_fd = (u_plus - u_minus) / (2 * eps)
    assert jnp.allclose(du_dE, du_dE_fd, rtol=1e-3, atol=1e-5)


def test_linear_elasticity_build_differentiable_solve():
    """Verify build_differentiable_bvp_solve computes reverse-mode gradients via JetSCI.

    BVP Problem Description:
    ------------------------
    - Physical Problem: 1D linear elastic rod under uniaxial tension with length L = 1.0.
    - Governing Differential Equation:
        d/dx [ E * A * du/dx ] = 0,   for x in (0, L)
    - Boundary Conditions:
        * Left end (x = 0, node 0): Fixed Dirichlet BC, u(0) = 0.0.
        * Right end (x = 1, node 2): Applied Neumann tensile point load, F = 10.0.
    - Material Parameter Vector:
        * Constitutive model: elastic_truss with parameter vector phi = [E, A].
        * Nominal values: E = 100.0, A = 1.0.
    - Objective Function:
        * Scalar loss corresponding to the tip displacement at x = L:
          Loss(phi) = u(L; phi) = (F * L) / (phi[0] * phi[1]) = 10.0 / (E * A)
    - Analytical Gradients:
        * dLoss / dE = -10.0 / (E^2 * A) = -0.001
        * dLoss / dA = -10.0 / (E * A^2) = -0.1

    Targeted Features:
    ------------------
    - High-Level API `build_differentiable_bvp_solve`: Verifies that `build_differentiable_bvp_solve`
      returns a JIT-compatible differentiable callable `solve_fn(phi, x0)` backed by JetSCI.
    - Reverse-Mode Automatic Differentiation (`jax.grad`): Validates that reverse-mode AD computes
      vector-Jacobian products (adjoint sensitivities) with respect to multi-dimensional
      material parameters phi = [E, A] in a single backward pass.
    - Verification: Compares computed gradient against exact analytical values.
    """
    # Discretize 1D domain [0, 1] with 2 linear interval elements
    points = np.linspace(0.0, 1.0, 3, dtype=np.float64).reshape((-1, 1))
    cells = np.array([[0, 1], [1, 2]], dtype=np.uint64)

    # Boundary conditions: fix left node (x=0), apply tension F=10.0 at right node (x=1)
    bcs = [
        DirichletBC(bc_type=BCType.NODE, component=0, index=0, value=0.0),
        NeumannBC(bc_type=BCType.NODE, component=0, index=2, value=10.0),
    ]

    fe_type = FiniteElementType(
        cell_type=CellType.interval,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=1,
            connectivity_en=cells,
            constitutive_model=elastic_truss,
            material_params=jnp.array([100.0, 1.0]),
        )
    ]

    # Build differentiable solve function wrapping JetSCI's differentiable solver
    solve_fn, phi0, x0, _, _ = build_differentiable_bvp_solve(
        vertices_vd=points,
        element_batches=element_batches,
        element_residual_func=linear_truss_residual,
        boundary_conditions=bcs,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        ),
    )

    # Scalar objective: tip displacement at x = 1 (last DOF)
    def loss_fn(phi):
        u = solve_fn(phi, x0)
        return u[-1]

    # Compute reverse-mode adjoint gradient w.r.t phi = [E, A]
    grad_ad = jax.grad(loss_fn)(phi0)

    # Verify against analytical gradient:
    # d(u_tip)/dE = -10 / (E^2 * A) = -0.001
    # d(u_tip)/dA = -10 / (E * A^2) = -0.1
    expected_grad = jnp.array([-0.001, -0.1])
    assert jnp.allclose(grad_ad, expected_grad, rtol=1e-4, atol=1e-6)


def test_nonlinear_hyperelasticity_differentiable_solve_bvp():
    """Verify solve_bvp is differentiable w.r.t constitutive parameters for nonlinear hyperelasticity.

    BVP Problem Description:
    ------------------------
    - Physical Problem: 2D plane strain hyperelastic solid on a unit square domain [0, 1] x [0, 1]
      subjected to finite deformation tension and shear.
    - Governing Differential Equation:
        div(P) = 0 in Omega, where P = F * S is the first Piola-Kirchhoff stress tensor,
        F = I + grad(u) is the deformation gradient, and E_gl = 0.5 * (F^T * F - I) is the
        Green-Lagrange strain tensor.
    - Constitutive Model:
        * St. Venant-Kirchhoff hyperelastic material model:
          S = lambda * tr(E_gl) * I + 2 * mu * E_gl
        * Lame parameters derived from Young's modulus E and Poisson's ratio nu:
          lambda = E * nu / ((1 + nu) * (1 - 2*nu)),  mu = E / (2 * (1 + nu))
        * Material parameters: E = 100.0, nu = 0.3.
    - Boundary Conditions:
        * Left boundary (x = 0): Nodes 0 and 3 are fully fixed (u_x = 0, u_y = 0).
        * Right boundary (x = 1):
          - Node 1: Prescribed displacement u_x = 0.05, u_y = 0.0.
          - Node 2: Prescribed displacement u_x = 0.05, free in y-direction.
    - FE Discretization:
        * Domain discretized into 2 linear triangular elements (P1 Lagrange basis, 4 vertices, 8 DOFs).
        * 2-point Gauss quadrature per element.

    Targeted Features:
    ------------------
    - Nonlinear BVP Solving via JetSCI: Verifies that `solve_bvp` executes JetSCI's Newton-Raphson
      solver (`NonlinearSolverType.JAX_NEWTON_RAPHSON`) with the CG linear solver to iteratively
      converge the nonlinear residual below tolerance.
    - Nonlinear Implicit Differentiation: Tests that `jax.jacobian` computes sensitivities
      of the converged displacement field with respect to Poisson's ratio (du/dnu) via the
      Implicit Function Theorem:
          du/dnu = - [d R / du]^{-1} [d R / dnu]
      evaluated at the converged equilibrium state.
    - Accuracy Verification: Compares autodiff Jacobian against central finite differences
      with perturbation step size eps = 1e-4.
    """
    # Define unit square geometry: 4 corner vertices discretized into 2 triangular elements
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)

    cells = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint64)

    # Boundary conditions: clamp left edge, displace right edge horizontally by 0.05
    bcs = [
        DirichletBC(bc_type=BCType.NODE, component=0, index=0, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=1, index=0, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=0, index=3, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=1, index=3, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=0, index=1, value=0.05),
        DirichletBC(bc_type=BCType.NODE, component=1, index=1, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=0, index=2, value=0.05),
    ]

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    # Initial constitutive parameters for 2 elements, Q quadrature points: [E = 100.0, nu = 0.3]
    init_params = jnp.full((2, 3, 2), 100.0).at[..., 1].set(0.3)

    base_batch = ElementBatch(
        fe_type=fe_type,
        n_dofs_per_basis=2,
        connectivity_en=cells,
        constitutive_model=st_venant_kirchhoff,
        material_params=init_params,
    )

    # Wrapper function mapping Poisson's ratio nu to converged displacement field u via solve_bvp
    def solve_nonlinear(nu_val):
        params = jnp.full((2, 3, 2), 100.0).at[..., 1].set(nu_val)
        batches = [base_batch.replace(material_params=params)]
        u, _, _ = solve_bvp(
            vertices_vd=points,
            element_batches=batches,
            element_residual_func=hyperelasticity_residual,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            ),
        )
        return u

    nu0 = 0.3
    u0 = solve_nonlinear(nu0)
    assert not jnp.any(jnp.isnan(u0))

    # Compute derivative du/dnu via forward-mode autodiff through Newton-Raphson solve
    du_dnu = jax.jacobian(solve_nonlinear)(nu0)

    # Verify against central finite differences
    eps = 1e-4
    u_plus = solve_nonlinear(nu0 + eps)
    u_minus = solve_nonlinear(nu0 - eps)
    du_dnu_fd = (u_plus - u_minus) / (2 * eps)

    assert jnp.allclose(du_dnu, du_dnu_fd, rtol=1e-3, atol=1e-5)


def test_nonlinear_unified_bvp_solver_differentiable():
    """Verify UnifiedBVPSolver build_material_parameter_solve produces differentiable solve for nonlinear problems.

    BVP Problem Description:
    ------------------------
    - Physical Problem: 2D nonlinear St. Venant-Kirchhoff hyperelastic body undergoing finite deformation
      on a unit square domain [0, 1] x [0, 1].
    - Governing Equations & Constitutive Model:
        * Nonlinear elastostatics: div(P) = 0.
        * St. Venant-Kirchhoff hyperelastic strain energy function with parameters [E, nu].
    - Boundary Conditions:
        * Clamped at x = 0 (nodes 0 and 3 fixed).
        * Displaced by Delta u_x = 0.05 at x = 1 (nodes 1 and 2).
    - FE Discretization:
        * 2 linear triangular elements (P1 Lagrange basis) with 8 total displacement DOFs.

    Targeted Features:
    ------------------
    - Object-Oriented Architecture `UnifiedBVPSolver`: Targets the `UnifiedBVPSolver` class
      and its method `build_material_parameter_solve()`.
    - Parameter Vector Mapping: Tests automatic flattening and unflattening of element batch
      constitutive parameters into a consolidated parameter vector phi.
    - Nonlinear Adjoint Sensitivity: Evaluates a scalar objective function (sum of displacements)
      and computes dObjective/dphi using reverse-mode automatic differentiation (`jax.grad`).
    - Verification: Compares the autodiff gradient component for Poisson's ratio against central
      finite differences.
    """
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)

    cells = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint64)

    bcs = [
        DirichletBC(bc_type=BCType.NODE, component=0, index=0, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=1, index=0, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=0, index=3, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=1, index=3, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=0, index=1, value=0.05),
        DirichletBC(bc_type=BCType.NODE, component=1, index=1, value=0.0),
        DirichletBC(bc_type=BCType.NODE, component=0, index=2, value=0.05),
    ]

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )

    init_params = jnp.full((2, 3, 2), 100.0).at[..., 1].set(0.3)

    batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=2,
            connectivity_en=cells,
            constitutive_model=st_venant_kirchhoff,
            material_params=init_params,
        )
    ]

    # Instantiate UnifiedBVPSolver configured with Newton-Raphson and CG linear solver
    solver = UnifiedBVPSolver(
        vertices_vd=points,
        element_batches=batches,
        element_residual_func=hyperelasticity_residual,
        boundary_conditions=bcs,
        options=UnifiedBVPOptions(
            solver_options=SolverOptions(
                nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
                linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            ),
        ),
    )

    # Build differentiable solve function mapping material parameter vector phi to displacement field u
    solve_phi, phi0, x0 = solver.build_material_parameter_solve()

    # Define scalar objective: sum of displacement components across all nodes
    def objective(phi):
        u = solve_phi(phi, x0)
        return jnp.sum(u)

    # Compute reverse-mode adjoint gradient w.r.t phi
    grad_ad = jax.grad(objective)(phi0)

    # Verify sensitivity along Poisson's ratio parameter (index 1) using central finite differences
    eps = 1e-4
    dphi = jnp.zeros_like(phi0).at[1].set(eps)
    f_plus = objective(phi0 + dphi)
    f_minus = objective(phi0 - dphi)
    grad_fd_1 = (f_plus - f_minus) / (2 * eps)

    assert jnp.allclose(grad_ad[1], grad_fd_1, rtol=1e-3, atol=1e-5)
