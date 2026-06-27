import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import buildKSP_Keith
import differentiateKSP_Keith


def jac(_x0):
    n = 8
    mat_shape = jnp.array([n, n], dtype=jnp.int64)
    mat_vals = jnp.array([4, 3, 9, 3, 4, 8, 6, 4], dtype=jnp.float64)
    mat_rows = jnp.arange(n, dtype=jnp.int32)
    mat_cols = jnp.arange(n, dtype=jnp.int32)
    return mat_shape, mat_vals, mat_rows, mat_cols


@jax.jit
def jitted_solve(handle, pc_handle, shape, rows, cols, vals, b):
    solver = buildKSP_Keith.__CupyCtx(handle=handle)
    pc = buildKSP_Keith.__CupyCtx(handle=pc_handle)
    return differentiateKSP_Keith.linearSolverSolve(solver, pc, shape, rows, cols, vals, b)


@jax.jit
def jitted_solve_loss(handle, pc_handle, shape, rows, cols, vals, b):
    solver = buildKSP_Keith.__CupyCtx(handle=handle)
    pc = buildKSP_Keith.__CupyCtx(handle=pc_handle)
    x = differentiateKSP_Keith.linearSolverSolve(solver, pc, shape, rows, cols, vals, b)
    return jnp.sum(x)


def lifecycle_inside_jit(x0):
    matrix = buildKSP_Keith.linearMatrixInit(jac=jac, x0=x0)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)
    try:
        shape, vals, rows, cols = jac(x0)
        b = jnp.ones((8,), dtype=jnp.float64)
        return differentiateKSP_Keith.linearSolverSolve(solver, pc, shape, rows, cols, vals, b)
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)


def main():
    shape, vals, rows, cols = jac(None)
    b = jnp.ones((8,), dtype=jnp.float64)

    matrix = buildKSP_Keith.linearMatrixInit(jac=jac, x0=None)
    pc = buildKSP_Keith.linearPCInit(matrix)
    solver = buildKSP_Keith.linearKSPInit(matrix)

    try:
        print("Testing jit around persistent-handle solve.")
        x = jitted_solve(solver.handle, pc.handle, shape, rows, cols, vals, b)
        print("jitted solve:", x)

        print("Testing grad of jitted persistent-handle loss wrt vals.")
        grad_vals = jax.grad(lambda v: jitted_solve_loss(solver.handle, pc.handle, shape, rows, cols, v, b))(vals)
        print("grad vals:", grad_vals)
    finally:
        buildKSP_Keith.linearSolverCleanup(solver)
        buildKSP_Keith.linearPCCleanup(pc)
        buildKSP_Keith.linearMatrixCleanup(matrix)

    print("Testing jit around lifecycle-containing function.")
    try:
        x0 = vals
        x_lifecycle = jax.jit(lifecycle_inside_jit)(x0)
        print("jitted lifecycle solve:", x_lifecycle)
    except Exception as err:
        print("jitted lifecycle function failed:")
        print(type(err).__name__, err)


if __name__ == "__main__":
    main()
