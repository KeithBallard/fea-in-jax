from helper import *

jax.config.update("jax_enable_x64", True)

import jax.extend
print(jax.extend.backend.get_backend().platform)


from jax.experimental import sparse
from petsc4py import PETSc

import time

import cupy as cp
from cupyx.scipy import sparse as cps

size = 10
iterations = 10

@jax.jit
def matmul(mat, vec):
    return sparse.coo_matvec(mat,vec)
    
    
vals = jnp.ones((size,),dtype=jnp.float64)*2
rows = jnp.arange(size,dtype=jnp.int64)
cols = jnp.arange(size,dtype=jnp.int64)

locations = jnp.asarray((rows,cols))
locations = locations.transpose()


mat = sparse.COO((vals,rows,cols),shape=(size,size),rows_sorted=True,cols_sorted=True)
vec = jnp.ones((size,),dtype=jnp.float64)

startTime = time.time()

for i in range(iterations):
    outputVec = matmul(mat,vec)
   
jax.block_until_ready(outputVec)

print("jax took", time.time()-startTime)
    
vec = jnp.ones((size,),dtype=jnp.float64)

mat_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
mat_petsc.setSizes([size,size])
mat_petsc.setType(PETSc.Mat.Type.AIJCUSPARSE)

mat_petsc.setPreallocationCOO(rows, cols)

mat_petsc.setValuesCOO(vals)

mat_petsc.assemblyBegin()
mat_petsc.assemblyEnd()

vec_petsc = PETSc.Vec().createWithArray(vec)
vec_petsc.setType(PETSc.Vec.Type.MPICUDA)

vec_petsc.assemblyBegin()
vec_petsc.assemblyEnd()

outputVec = vec_petsc.duplicate()


mat_petsc.view()

startTime = time.time()

for i in range(iterations):
    mat_petsc.mult(vec_petsc,outputVec)
    
print("petsc took", time.time()-startTime)


vals = cp.ones((size,1),dtype=cp.float64)*2
rows = cp.arange(size,dtype=cp.int64)
cols = cp.arange(size,dtype=cp.int64)
cp_vec = cp.ones((size,1),dtype=cp.float64)


vals = vals.flatten()
rows = rows.flatten()
cols = cols.flatten()
cp_vec = cp_vec.flatten()




cp_mat = cps.coo_matrix((vals,(rows,cols)),shape=(size,size))




startTime = time.time()

for i in range(iterations):
    cp_mat.dot(cp_vec)
    
print("cupy took", time.time()-startTime)