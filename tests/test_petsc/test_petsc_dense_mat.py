from petsc4py import PETSc
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from helper import *
import time
import jax.dlpack as jdl
import cupy as cp

#jax.config.update("jax_platform_name", "cpu")

# TODO write note about backend
float_byte_size = 4
if PETSc.RealType == np.float64:
    jax.config.update("jax_enable_x64", True)
    float_byte_size = 8
elif PETSc.RealType == np.float32:
    pass
else:
    raise Exception(
        f"Floating point types are incompatible between PETSc ({PETSc.RealType}) and JAX."
    )


def to_petsc(array: jnp.array):
    """
    TODO document
    """

    assert len(array.devices()) == 1, "Sharded arrays are not supported"
    assert len(array.shape) <= 2, "Arrays with dimensionality > 2 are not supported"

    if len(array.shape) == 1:
        array_p = PETSc.Vec()
        array_p.createWithDLPack(jdl.to_dlpack(array))
        return array_p

    else:  # len(array.shape) == 2
        if "CPU" in str(array.device):
            array_p = PETSc.Mat()
            array_p.createDense(size=array.shape, array=array)
            return array_p
        elif "cuda" in str(array.device):
            assert array.shape[0] == array.shape[1], "Array must be square"
            array_p = PETSc.Mat()
            array_p.createDenseCUDA(size=array.shape[0], array=None, cudahandle=array.unsafe_buffer_pointer())
            return array_p
        else:
            raise Exception(f'Device {array.device} is not supported')

print('Before allocate', get_current_pid_host_memory())
A = jnp.ones(shape=(20_000, 20_000))
print('After allocate', get_current_pid_host_memory())
print(f"Expected size of array: {A.shape[0] * A.shape[1] * float_byte_size / 1e6} MB")
A_p = to_petsc(A)
print('After to_petsc', get_current_pid_host_memory())
print(A_p)
print('After print', get_current_pid_host_memory())

#v = PETSc.Viewer()
#v(A_p)
