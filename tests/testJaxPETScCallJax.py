import numpy as np

import jax
import jax.numpy as jnp
from petsc4py import PETSc

import CreateJaxMat


jax.config.update("jax_enable_x64", True)


def _petsc_solve_calling_jax_impl(diag, b):
    diag = jnp.asarray(diag)
    b = jnp.asarray(b)
    n = b.shape[0]
    calls = {"mult": 0}

    @jax.jit
    def jacobian_mult(x):
        return jnp.asarray(diag) * x

    def counted_jacobian_mult(x):
        calls["mult"] += 1
        y = jacobian_mult(x)
        y.block_until_ready()
        return y

    mat, _ = CreateJaxMat.buildPETScJaxMat(
        shape=(n, n),
        jacobianFunc=counted_jacobian_mult,
        comm=PETSc.COMM_SELF,
    )
    ksp = CreateJaxMat.buildPETScKSP(
        mat,
        solverType=PETSc.KSP.Type.GMRES,
        precondType=PETSc.PC.Type.NONE,
        comm=PETSc.COMM_SELF,
    )
    ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=100)

    b_petsc = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    b_petsc.getArray(readonly=False)[...] = b
    x_petsc = b_petsc.duplicate()
    x_petsc.set(0.0)

    ksp.solve(b_petsc, x_petsc)
    x = np.asarray(x_petsc.getArray()).copy()

    print("PETSc converged reason:", ksp.getConvergedReason())
    print("PETSc iterations:", ksp.getIterationNumber())
    print("JaxMat.mult calls:", calls["mult"])

    x_petsc.destroy()
    b_petsc.destroy()
    ksp.destroy()
    mat.destroy()

    return x


def petsc_solve_calling_jax(diag, b):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(_petsc_solve_calling_jax_impl, result_info, diag, b)


def main():
    diag = jnp.array([4.0, 3.0, 9.0, 3.0, 4.0, 8.0, 6.0, 4.0], dtype=jnp.float64)
    b = jnp.ones_like(diag)
    expected = b / diag

    x = petsc_solve_calling_jax(diag, b)
    x.block_until_ready()

    print("x:", x)
    print("expected:", expected)
    np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    main()
