from petsc4py import PETSc
import jax
import jax.numpy as jnp
import numpy as np

import sys
import os.path

# Adding parent directory for helper
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from helper import *

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

N = 60_000_000
E = N - 1
cells = np.zeros((E, 2), dtype=np.uint64)
cells[:, 0] = np.arange(E, dtype=np.uint64)
cells[:, 1] = np.arange(E, dtype=np.uint64) + 1
assert cells.shape == (E, 2)
print(cells)

# Create row offset map for the compressed sparse row (CSR) format
print('Before CSR allocation', get_current_pid_host_memory())
csr_row_ind = np.array(build_row_ind(N, cells, np.int32), dtype=np.int32)
assert csr_row_ind.shape == (N + 1,)
csr_col_ind, csr_data = build_col_ind_and_data(csr_row_ind, cells, np.int32)
csr_data = np.array(csr_data, dtype=np.float64)
print('After CSR allocation', get_current_pid_host_memory())
csr_data *= 2.0
print(f'csr_row_ind {csr_row_ind.dtype}: {csr_row_ind}')
print(f'csr_col_ind {csr_col_ind.dtype}: {csr_col_ind}')
print(f'csr_data {csr_data.dtype}: {csr_data}')

nnz = csr_col_ind.shape[0]
A = PETSc.Mat().create()
A.setType('aijcusparse')
A.setSizes([csr_row_ind.shape[0] - 1, csr_row_ind[-1]])
A.setPreallocationCSR(csr=(csr_row_ind, csr_col_ind))
A.setUp()
print(A)

A.setValuesCSR(I=csr_row_ind, J=csr_col_ind, V=csr_data)
A.assemble()

csr_data *= 2.0
print(f'csr_data {csr_data.dtype}: {csr_data}')

# Note: A will not reflect changes to csr_data
#v = PETSc.Viewer()
#v(A)

csr_row_ind_jax = jnp.array(csr_row_ind, dtype=jnp.int64)
csr_col_ind_jax = jnp.array(csr_col_ind, dtype=jnp.int64)
csr_data_jax = jnp.array(csr_data, dtype=jnp.float64)
print(f'csr_row_ind_jax {type(csr_row_ind_jax)} {csr_row_ind_jax.device}')
print(f'csr_col_ind_jax {type(csr_col_ind_jax)} {csr_col_ind_jax.device}')
print(f'csr_data_jax {type(csr_data_jax)} {csr_data_jax.device}')

print(f'Expected memory for CSR: {(csr_row_ind_jax.shape[0] * 8 + csr_col_ind_jax.shape[0] * 8 + csr_data_jax.shape[0] * 8) / 1e6}')
mem1 = get_current_pid_host_memory()
print('Before setValuesCSR with CUDA arrays', mem1)
# Note that the JAX arrays can be passed but it will (probably) copy the JAX array to host
# then copy the data to CUDA buffers in the matrix.
A.setValuesCSR(I=csr_row_ind_jax, J=csr_col_ind_jax, V=csr_data_jax)
A.assemble()
mem2 = get_current_pid_host_memory()
print('After setValuesCSR with CUDA arrays', mem2)
print(f'Difference: {mem2 - mem1}')

#v = PETSc.Viewer()
#v(A)