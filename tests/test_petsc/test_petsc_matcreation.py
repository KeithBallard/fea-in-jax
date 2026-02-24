import petsc4py
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix

import jax.numpy as jnp
from jax import random
from jax.experimental import sparse


M = 100
N = 100

testVector = jnp.ones((100,1),dtype=jnp.float32)

key = random.key(1998)
randomValue = random.normal(key,(1000),dtype=jnp.float32)


mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
mat.setSizes(((PETSc.DECIDE,M),(PETSc.DECIDE,N)))
mat.setType(PETSc.Mat.Type.AIJ)

mat.setPreallocationNNZ((5,5))

istart, iend = mat.getOwnershipRange()

for i in range(istart,iend):
    cols = jnp.array([i],dtype=jnp.int16)
    vals = randomValue[i]
    mat.setValues(i,cols,vals)

mat.assemblyBegin()
mat.assemblyEnd()


vec = PETSc.Vec().createWithArray(testVector)
#vec = PETSc.Vec().createWithDLPack(testVector) causes some kind of mismatch so that implies we need to do some modifications 

x = vec.duplicate()

ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setOperators(mat)
ksp.setType('cg')

ksp.solve(vec,x)

print(x.getArray())