"""Shape/layout smoke tests for PETSc Vec <-> JAX conversion.

This intentionally ignores MPI/distribution and ownership-origin questions.
The goal is only to check how simple local vector shapes should be represented
when PETSc is the caller and JAX functions expect structured data.

Run:

    /home/alberto/venvs/mpi-gpu/bin/python v10/test_snes_vec_shape_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from petsc4py import PETSc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10.NonlinearSNES import (
    convertJaxMatToCOOData,
    jaxArrayToPETScVec,
    petscVecToJAX,
)


jax.config.update("jax_enable_x64", True)


def print_array(name, value):
    print(f"{name}: shape={value.shape}, dtype={value.dtype}")
    print(value)


def create_petsc_cuda_vec(size):
    vec = PETSc.Vec().create(PETSc.COMM_SELF)
    vec.setSizes(size)
    try:
        vec.setType(PETSc.Vec.Type.CUDA)
    except AttributeError:
        vec.setType("cuda")
    vec.setUp()
    return vec


def exampleJaxVecChange(x):
    return 15.0               #this should set all values to 15

exampleJaxVecChange = jax.vmap(exampleJaxVecChange)
exampleJaxVecChange = jax.jit(exampleJaxVecChange)

def examplePETScVecChange(vec):
    vec.setValues([0, 1, 2, 3, 4, 5], [12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
    vec.assemble()


def check_flat_JAX_vec_roundtrip():
    print("\n=== flat vector roundtrip ===")
    x = jnp.arange(6, dtype=jnp.float64) + 1.0
    vec = jaxArrayToPETScVec(x)
    try:
        vec.view()
        x_back = petscVecToJAX(vec)
        print_array("x", x)
        print_array("x_back", x_back)
        assert x_back.shape == (6,)
        assert jnp.allclose(x_back, x)
    finally:
        vec.destroy()

def check_flat_PETSc_vec_roundtrip():
    print("\n=== flat vector roundtrip other direction ===")
    vec = create_petsc_cuda_vec(6)
    vec.setValues([0, 1, 2, 3, 4, 5], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    vec.assemble()
    vec_back = None
    try:
        x = petscVecToJAX(vec)
        vec_back = jaxArrayToPETScVec(x)
        print_array("x", x)
        vec_back.view()
        x_roundtrip = petscVecToJAX(vec_back)
        print_array("x_roundtrip", x_roundtrip)
        assert x.shape == (6,)
        assert vec_back.getSize() == 6
        assert x_roundtrip.shape == (6,)
        assert jnp.allclose(x_roundtrip, x)
    finally:
        vec.destroy()
        if vec_back is not None:
            vec_back.destroy()



def check_Jax_roundtrip_valChange():
    print("\n=== JAX-created Vec mutated by PETSc before JAX readback ===")
    x = jnp.arange(6, dtype=jnp.float64) + 1.0
    vec = jaxArrayToPETScVec(x)
    try:
        vec.view()
        examplePETScVecChange(vec)
        vec.view()
        print_array("x", x)
        x_back = petscVecToJAX(vec)
        print_array("x_back", x_back)
        expected = jnp.full((6,), 12.0, dtype=jnp.float64)
        assert x_back.shape == (6,)
        assert jnp.allclose(x_back, expected)
    finally:
        vec.destroy()


def check_PETSc_roundtrip_valChange():
    print("\n=== PETSc-created Vec changed by JAX before PETSc sendback ===")
    vec = create_petsc_cuda_vec(6)
    vec.setValues([0, 1, 2, 3, 4, 5], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    vec.assemble()
    vec_back = None
    try:
        x = petscVecToJAX(vec)
        x = exampleJaxVecChange(x)
        print_array("x", x)
        vec.view()
        vec_back = jaxArrayToPETScVec(x)
        vec_back.view()
        x_roundtrip = petscVecToJAX(vec_back)
        print_array("x_roundtrip", x_roundtrip)
        expected = jnp.full((6,), 15.0, dtype=jnp.float64)
        assert x_roundtrip.shape == (6,)
        assert vec_back.getSize() == 6
        assert jnp.allclose(x_roundtrip, expected)
    finally:
        vec.destroy()
        if vec_back is not None:
            vec_back.destroy()


def check_rank2_as_flat_vec_roundtrip():
    print("\n=== rank-2 data represented as flat PETSc Vec ===")
    logical_shape = (2, 3)
    x_matrix = jnp.arange(6, dtype=jnp.float64).reshape(logical_shape) + 1.0
    vec = jaxArrayToPETScVec(x_matrix.reshape(-1))
    try:
        x_flat_back = petscVecToJAX(vec)
        x_matrix_back = x_flat_back.reshape(logical_shape)
        print_array("x_matrix", x_matrix)
        print_array("x_flat_back", x_flat_back)
        print_array("x_matrix_back", x_matrix_back)
        assert x_flat_back.shape == (6,)
        assert x_matrix_back.shape == logical_shape
        assert jnp.allclose(x_matrix_back, x_matrix)
    finally:
        vec.destroy()


def check_rank2_direct_vec_creation_diagnostic():
    print("\n=== direct rank-2 JAX array -> PETSc Vec diagnostic ===")
    x_matrix = jnp.arange(6, dtype=jnp.float64).reshape((2, 3)) + 1.0
    vec = None
    try:
        vec = jaxArrayToPETScVec(x_matrix)
        x_back = petscVecToJAX(vec)
        print("direct rank-2 createWithDLPack succeeded")
        print_array("x_back", x_back)
        print(
            "diagnostic: PETSc Vec should still be treated as logically flat; "
            "reshape explicitly in JAX user functions."
        )
    except Exception as exc:
        print("direct rank-2 createWithDLPack failed, which is acceptable for this test:")
        print(type(exc).__name__, exc)
    finally:
        if vec is not None:
            vec.destroy()


def structured_residual_from_flat(x_flat, logical_shape):
    x = x_flat.reshape(logical_shape)
    residual = jnp.stack(
        [
            x[0, 0] ** 2 + 2.0 * x[0, 1] - x[1, 2],
            x[0, 1] * x[1, 0] + jnp.sin(x[0, 2]),
            x[0, 2] ** 2 - x[1, 1],
            x[1, 0] + 3.0 * x[1, 1],
            x[1, 1] * x[1, 2],
            x[1, 2] ** 3 - x[0, 0],
        ]
    )
    return residual


def check_structured_jax_function_from_petsc_vec():
    print("\n=== PETSc flat Vec -> structured JAX residual/Jacobian ===")
    logical_shape = (2, 3)
    x_matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64)
    x_flat = x_matrix.reshape(-1)
    vec = jaxArrayToPETScVec(x_flat)
    try:
        x_from_petsc = petscVecToJAX(vec)
        residual = structured_residual_from_flat(x_from_petsc, logical_shape)
        jac = jax.jacfwd(lambda z: structured_residual_from_flat(z, logical_shape))(x_from_petsc)
        coo = convertJaxMatToCOOData(jac)

        print_array("x_from_petsc", x_from_petsc)
        print_array("residual", residual)
        print_array("dense jacobian", jac)
        print("COO shape:", coo.shape)
        print("COO vals shape:", coo.vals.shape)
        print("COO rows first 12:", coo.rows[:12])
        print("COO cols first 12:", coo.cols[:12])

        assert x_from_petsc.shape == (6,)
        assert residual.shape == (6,)
        assert jac.shape == (6, 6)
        assert coo.shape.tolist() == [6, 6]
        assert coo.vals.shape == (36,)
    finally:
        vec.destroy()


def check_block_size_metadata_does_not_define_jax_shape():
    print("\n=== PETSc block size metadata diagnostic ===")
    x = jnp.arange(6, dtype=jnp.float64) + 1.0
    vec = jaxArrayToPETScVec(x)
    try:
        vec.setBlockSize(3)
        x_back = petscVecToJAX(vec)
        print("PETSc block size:", vec.getBlockSize())
        print_array("x_back", x_back)
        print("diagnostic: block size is PETSc metadata; JAX still sees a flat vector.")
        assert x_back.shape == (6,)
    finally:
        vec.destroy()


def main():
    check_flat_JAX_vec_roundtrip()
    check_flat_PETSc_vec_roundtrip()
    check_Jax_roundtrip_valChange()
    check_PETSc_roundtrip_valChange()
    check_rank2_as_flat_vec_roundtrip()
    check_rank2_direct_vec_creation_diagnostic()
    check_structured_jax_function_from_petsc_vec()
    check_block_size_metadata_does_not_define_jax_shape()
    print("\nAll shape/layout smoke checks completed.")


if __name__ == "__main__":
    main()
