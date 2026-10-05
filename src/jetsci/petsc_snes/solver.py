from dataclasses import dataclass

from petsc4py import PETSc

from ..conversions import *
from ..options import *

@dataclass
class PETScLinearSolver:
    """Owns the PETSc KSP object to be used in automatic differentiation.

    This is the linear-solve companion to `PETScNonlinearSolver`. It owns the
    live PETSc KSP plus its work vectors and operator matrix so that the IFT
    derivative solve can reuse PETSc directly instead of falling back to JAX.
    """
    ksp: object
    vector_callback: Callable
    matrix_callback: Callable
    options: SolverOptions
    vector_data: object | None = None
    matrix_data: object | None = None
    operator_matrix: object | None = None


    def __post_init__(self):
        """Setup the KSP work objects."""
        self.vector_data = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.matrix_data = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self.matrix_data.setType('aijcusparse')
        self.operator_matrix = self.matrix_data
        self.ksp.setOperators(self.operator_matrix, self.operator_matrix)


    def _ensure_size(self, x0: jnp.ndarray):
        """Ensure the vector and matrix storage are the correct size."""
        if self.vector_data.getType() is None:
            self.vector_data.setType("cuda")
            self.vector_data.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.vector_data.setUp()
        if self.matrix_data.getType() is None:
            self.matrix_data.setSizes((x0.shape[0], x0.shape[0]))
            self.matrix_data.setUp()
            self.operator_matrix = self.matrix_data
            self.ksp.setOperators(self.operator_matrix, self.operator_matrix)


    def update_operator(self, operator_matrix):
        """Replace the linear operator used by the KSP."""
        if self.matrix_data is not None and self.matrix_data is not operator_matrix:
            self.matrix_data.destroy()
        self.operator_matrix = operator_matrix
        self.matrix_data = operator_matrix
        self.ksp.setOperators(operator_matrix, operator_matrix)
        return self


    def linear_solve(self, rhs: jnp.ndarray):
        """Solve with this KSP object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """
        self._ensure_size(rhs)

        rhs_vec = jax_array_to_petsc_vec(rhs)
        x = rhs_vec.duplicate()
        try:
            rhs_vec.copy(x)
            self.ksp.solve(rhs_vec, x)
            return x
        finally:
            rhs_vec.destroy()

    def block_linear_solve(self,rhs_block):


        output_block = rhs_block.duplicate()
        self.ksp.matSolve(rhs_block,output_block)

        return output_block

    def transpose_linear_solve(self, rhs: jnp.ndarray):
        """Solve adjoint problem with this KSP object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """
        self._ensure_size(rhs)

        rhs_vec = jax_array_to_petsc_vec(rhs)
        x = rhs_vec.duplicate()
        try:
            rhs_vec.copy(x)
            self.ksp.solveTranspose(rhs_vec, x)
            return x
        finally:
            rhs_vec.destroy()


    def solve(self, rhs: jnp.ndarray):
        """Alias for `linear_solve`."""
        return self.linear_solve(rhs)
    
    def solve_transpose(self, rhs: jnp.ndarray):
        """Alias for `transpose_linear_solve`."""
        return self.transpose_linear_solve(rhs)


    def solve_to_jax(self, rhs):
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""
        x = self.linear_solve(rhs)
        try:
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()
            return result
        finally:
            x.destroy()

    def solve_transpose_to_jax(self, rhs):
        """Solve transpose and explicitly copy the PETSc Vec result into a JAX array."""
        x = self.transpose_linear_solve(rhs)
        try:
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()
            return result
        finally:
            x.destroy()

    def cleanup_work_vectors(self):
        """Destroy work objects that depend on vector size."""
        if self.vector_data is not None:
            self.vector_data.destroy()
            self.vector_data = None
        if self.matrix_data is not None:
            self.matrix_data.destroy()
            self.matrix_data = None
        self.operator_matrix = None

    def destroy(self):
        """Destroy all PETSc objects owned by this wrapper."""
        self.cleanup_work_vectors()
        self.ksp.destroy()

@dataclass
class PETScNonlinearSolver:
    """Own the PETSc SNES object and its callback companion objects."""

    snes: object
    residual_callback: Callable
    jacobian_callback: Callable
    options: SolverOptions
    residual_vec: object | None = None
    jacobian_mat: object | None = None
    jacobian_callback_state: object | None = None
    callback_stats: dict | None = None
    diagnostics: bool = False
    last_diagnostics: dict | None = None
    last_ksp_residual_history: list[tuple[int, float]] | None = None

    def __post_init__(self):
        """Setup snes and Mat/Vec."""
        self.residual_vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.jacobian_mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self.jacobian_mat.setType('aijcusparse')
        self.snes.setFunction(self.residual_callback, self.residual_vec)
        self.snes.setJacobian(self.jacobian_callback, self.jacobian_mat, self.jacobian_mat)


    def _ensure_size(self, x0: jnp.ndarray):
        """Ensure the vector is the correct size"""
        if self.residual_vec.getType() is None:
            self.residual_vec.setType("cuda")
            self.residual_vec.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.residual_vec.setUp()


    def solve(self, x0: jnp.ndarray):
        """Solve with this SNES object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """
        solve_start = perf_counter()
        self._ensure_size(x0)

        conversion_start = perf_counter()
        x0_vec = jax_array_to_petsc_vec(x0)
        conversion_time = perf_counter() - conversion_start
        x = x0_vec.duplicate()
        try:
            copy_start = perf_counter()
            x0_vec.copy(x)
            copy_time = perf_counter() - copy_start
            ksp = self.snes.getKSP()
            ksp_residual_history = []

            def ksp_monitor(ksp, iteration, residual_norm):
                del ksp
                ksp_residual_history.append((int(iteration), float(residual_norm)))

            if hasattr(ksp, "cancelMonitor"):
                ksp.cancelMonitor()
            ksp.setMonitor(ksp_monitor)
            petsc_start = perf_counter()
            self.snes.solve(None, x)
            petsc_time = perf_counter() - petsc_start
            self.last_ksp_residual_history = ksp_residual_history
            update_vec = x.duplicate()
            try:
                x.copy(update_vec)
                update_vec.axpy(-1.0, x0_vec)
                solution_update_norm = update_vec.norm()
            finally:
                update_vec.destroy()
            self.last_diagnostics = {
                "total_s": perf_counter() - solve_start,
                "input_conversion_s": conversion_time,
                "initial_copy_s": copy_time,
                "snes_solve_s": petsc_time,
                "snes_iterations": self.snes.getIterationNumber(),
                "snes_converged_reason": self.snes.getConvergedReason(),
                "snes_function_norm": self.snes.getFunctionNorm(),
                "snes_solution_norm": x.norm(),
                "snes_solution_update_norm": solution_update_norm,
                "snes_ksp_iterations": ksp.getIterationNumber(),
                "snes_ksp_converged_reason": ksp.getConvergedReason(),
                "snes_ksp_residual_norm": ksp.getResidualNorm(),
                "snes_ksp_tolerances": ksp.getTolerances(),
                "snes_ksp_norm_type": ksp.getNormType() if hasattr(ksp, "getNormType") else None,
                "snes_ksp_residual_history_count": len(ksp_residual_history),
                "snes_ksp_residual_history_head": ksp_residual_history[:8],
                "snes_ksp_residual_history_tail": ksp_residual_history[-8:],
                "callback_stats": dict(self.callback_stats or {}),
            }
            if self.diagnostics:
                print("PETSc SNES diagnostics:", self.last_diagnostics)
            return x
        finally:
            x0_vec.destroy()

    def solve_to_jax(self, x0):
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""
        x = self.solve(x0)
        try:
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()
            return result
        finally:
            x.destroy()

    def linear_solve(self, x0):
        pass

    def cleanup_work_vectors(self):
        """Destroy residual/Jacobian objects that depend on vector size."""
        if self.residual_vec is not None:
            self.residual_vec.destroy()
            self.residual_vec = None
        if self.jacobian_mat is not None:
            self.jacobian_mat.destroy()
            self.jacobian_mat = None

    def destroy(self):
        """Destroy all PETSc objects owned by this wrapper."""
        self.cleanup_work_vectors()
        self.snes.destroy()


def validate_petsc_solver_options(options: SolverOptions) -> None:
    """Validate that selected solver families use compatible method enums."""
    if options.nonlinear_solver_type is NonlinearSolverType.PETSC_SNES:
        if not isinstance(options.linear_solve_type, PETScLinearSolverType):
            raise TypeError(
                "PETSc SNES requires a PETSc linear solver method. "
                f"Got {options.linear_solve_type!r}."
            )
        if not isinstance(options.linear_precond_type, PETScPreconditionerType):
            raise TypeError(
                "PETSc SNES requires a PETSc preconditioner method. "
                f"Got {options.linear_precond_type!r}."
            )
