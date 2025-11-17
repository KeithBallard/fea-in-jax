from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix

import jax.numpy as jnp

A_sp = csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
indptr = jnp.array(A_sp.indptr)
indices = jnp.array(A_sp.indices)
data = jnp.array(A_sp.data)

print(indptr.devices())

A = PETSc.Mat()
A.create(comm=PETSc.COMM_WORLD) # Or PETSc.COMM_SELF for a local matrix
A.setSizes([A_sp.shape[0], A_sp.shape[1]])
A.setType(PETSc.Mat.Type.AIJCUSPARSE) # This specifies the CUDA sparse matrix type

A.createAIJWithArrays(size=(A_sp.shape[0], A_sp.shape[1]), csr=(indptr, indices, data))


A.assemblyBegin()
A.assemblyEnd()

A.view()
