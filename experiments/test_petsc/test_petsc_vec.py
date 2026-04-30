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


start_mem = get_current_pid_host_memory()
print(f"Pre allocate mem: {start_mem}")

x = jnp.ones(shape=(400_000_000,))
x.block_until_ready()
# x = np.ones(shape=(40_000_000,))
print(x.devices())
print(x.device)
exit()

alloc_mem = get_current_pid_host_memory()
print(f"After allocation mem: {alloc_mem}")
print(f"Allocation was: {alloc_mem - start_mem}")
print(f"Expected size of array: {x.shape[0] * float_byte_size / 1e6} MB")
print(
    f"Unexpected extra memory allocated: {alloc_mem - start_mem - x.shape[0] * float_byte_size / 1e6} MB"
)


print("Pre petsc mem:", get_current_pid_host_memory())
x_petsc = PETSc.Vec()
# NOTE copies to host if x is on device
x_petsc.createWithArray(x)
# NOTE const is thrown away!
x_petsc.setValues(0, 2.0)
print("Post petsc mem:", get_current_pid_host_memory())


x2_petsc = PETSc.Vec()
# NOTE passes universal buffer (avoiding host copy), requires PETSc with CUDA
x2_petsc.createWithDLPack(jdl.to_dlpack(x))

print("Post petsc 2 mem:", get_current_pid_host_memory())

print(x_petsc.array.data)
print(x_petsc.array_r.data)
print(x_petsc.array_w.data)

x2_petsc.setValues(0, 2.0)

print("Post print mem:", get_current_pid_host_memory())

v = PETSc.Viewer()
# v(x_petsc)
