import petsc4py
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix

import jax.numpy as jnp

A_sp = csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
indptr = jnp.array(A_sp.indptr)
indices = jnp.array(A_sp.indices)
data = jnp.array(A_sp.data)

"""print(indptr.devices())

A = PETSc.Mat()
A.create(comm=PETSc.COMM_WORLD) # Or PETSc.COMM_SELF for a local matrix
A.setSizes([A_sp.shape[0], A_sp.shape[1]])
A.setType(PETSc.Mat.Type.AIJ) # This specifies the CUDA sparse matrix type

A.createAIJWithArrays(size=(A_sp.shape[0], A_sp.shape[1]), csr=(indptr, indices, data))


A.assemblyBegin()
A.assemblyEnd()

A.view()


"""

problemSize = A_sp.shape

mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
mat.setSizes(problemSize)
mat.setType(PETSc.Mat.Type.AIJ) #set type to AIJCUSPARSE then assemble? Maybe just ask the PETSc devs


mat.setValuesCSR(indptr,indices,data)
                                                                #figure out if it's better to build the matrix like this or like it's done in linear solve?
mat.assemblyBegin()
mat.assemblyEnd()

mat.view()

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src") #every time python imports from a parent directory an angel weeps

import jax
print(jax.devices())


def get_petsc_location():
    """Prints the PETSc installation directory used by petsc4py."""
    config = petsc4py.get_config()
    if 'PETSC_DIR' in config:
        petsc_dir = config['PETSC_DIR']
        petsc_arch = config['PETSC_ARCH']
        print(f"PETSc installation directory: {petsc_dir}")
        print(config)
        return petsc_dir
    else:
        # Fallback to environment variables, which might not be set in all cases
        petsc_dir = os.environ.get('PETSC_DIR')
        if petsc_dir:
            print(f"PETSc installation directory (from env var): {petsc_dir}")
            return petsc_dir
        else:
            print("Could not automatically determine PETSc installation directory.")
            return None












from fe_jax import sparse_linear_solve

import jax.experimental.sparse as jsp

jaxCOO = jsp.coo_fromdense(A_sp.todense())

testVec = jnp.array([1,2,3])

matDictVal = sparse_linear_solve.__petsc_init(jaxCOO)
solutionObject = sparse_linear_solve.__petsc_solve(matDictVal,testVec)
print(solutionObject)





