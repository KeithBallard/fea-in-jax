import petsc4py
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix

from scipy.sparse.linalg import LaplacianNd

import jax

import jax.numpy as jnp
from jax import random
from jax.experimental import sparse

import cupy as cp


import sys
import os.path

import matplotlib.pyplot as plt

testPreallocate1 = False
testPreallocate2 = False
testPreallocate3 = False
testPreallocate4 = False

M = 100
N = 100

testVector = jnp.ones((100,1),dtype=jnp.float32)

key = random.key(1998)
randomValue = random.normal(key,(1000),dtype=jnp.float32)

if testPreallocate1:
    mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    mat.setSizes(((PETSc.DECIDE,M),(PETSc.DECIDE,N)))
    mat.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)

    mat.setPreallocationNNZ((5,5))

    istart, iend = mat.getOwnershipRange()

    for i in range(istart,iend):
        cols = jnp.array([i],dtype=jnp.int16)
        vals = randomValue[i]
        mat.setValues(i,cols,vals)

    mat.assemblyBegin()
    mat.assemblyEnd()

    vec = PETSc.Vec().createWithArray(testVector)
    vec.setType(PETSc.Vec.Type.MPICUDA)
    #vec = PETSc.Vec().createWithDLPack(testVector) causes some kind of mismatch so that implies we need to do some modifications 

    x = vec.duplicate()

    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(mat)
    ksp.setType('cg')

    ksp.solve(vec,x)

    print(x.getArray())   #this may have a copy off gpu to cpu back to gpu
    print("Works, but involves construction on the CPU")

if testPreallocate2:

    rows = cp.array([0,1,5], dtype=cp.int32)
    cols = cp.array([1,2,5], dtype=cp.int32)

    comm = PETSc.COMM_SELF
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([M, N])
    A.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)

    A.setPreallocationCOO(rows.get(), cols.get()) #this causes an error because there is currently no API for reading non-host side arrays into matrix creation

    # Values also on GPU
    vals = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32) #oddly this does not despite the fact the jnp array should live on GPU. is there an implicit transfer? Or is it somehow sidestepping cuda_array_interface
    A.setValuesCOO(vals)

    A.assemblyBegin()
    A.assemblyEnd()

    print(A.view())

if testPreallocate3:
    rows = jnp.arange(100,dtype=jnp.int32)
    cols = jnp.arange(100,dtype=jnp.int32)

    comm = PETSc.COMM_SELF
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([M, N])
    A.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)

    A.setPreallocationCOO(rows, cols)

    vals = jnp.ones((100,1),dtype=jnp.float32)

    A.setValuesCOO(vals)

    A.assemblyBegin()
    A.assemblyEnd()

    vec = PETSc.Vec().createWithArray(testVector)
    vec.setType(PETSc.Vec.Type.MPICUDA)
    #vec = PETSc.Vec().createWithDLPack(testVector) causes some kind of mismatch so that implies we need to do some modifications 

    x = vec.duplicate()

    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType('cg')

    ksp.solve(vec,x)

    print(x.getArray())


if testPreallocate3:
    rows = jnp.arange(100,dtype=jnp.int32)
    cols = jnp.arange(100,dtype=jnp.int32)

    comm = PETSc.COMM_SELF
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([M, N])
    A.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)

    A.setPreallocationCOO(rows, cols)

    vals = random.normal(key,(100,1),dtype=jnp.float32)

    A.setValuesCOO(vals)

    A.assemblyBegin()
    A.assemblyEnd()

    vec = PETSc.Vec().createWithArray(testVector)
    vec.setType(PETSc.Vec.Type.MPICUDA)
    #vec = PETSc.Vec().createWithDLPack(testVector) causes some kind of mismatch so that implies we need to do some modifications 

    x = vec.duplicate()
    print(x.getArray())

    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType('cg')

    ksp.solve(vec,x)

    print(x.getArray())

if testPreallocate4:

    rows = cp.arange(100,dtype=jnp.int32)
    cols = cp.arange(100,dtype=jnp.int32)
    vals = jnp.ones((100,1),dtype=jnp.float32)

    comm = PETSc.COMM_SELF

    rows = rows.astype(cp.int32, copy=False)
    cols = cols.astype(cp.int32, copy=False)

    mat = PETSc.Mat().create(comm=comm)
    mat.setType(PETSc.Mat.Type.AIJCUSPARSE)
    mat.setSizes([M, N])

    mat.setPreallocationCOO(rows.get(), cols.get())
    mat.setValuesCOO(vals)   # one safe host copy








rows = jnp.arange(100,dtype=jnp.int32)
cols = jnp.arange(100,dtype=jnp.int32)
comm = PETSc.COMM_SELF
mat = PETSc.Mat().create(comm=comm)
mat.setSizes([M, N])
mat.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)
mat.setPreallocationCOO(rows, cols)


gpu_coo_array = cp.ones((100,1), dtype=cp.float64)*4

import ctypes as ct
lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.
mat_ptr = ct.c_void_p(mat.handle)  # the low level pointer of the mat object
coo_ptr = ct.c_void_p(gpu_coo_array.data.ptr)  # the pointer to GPU memory


MatSetValuesCOO(mat_ptr, coo_ptr, PETSc.InsertMode.INSERT_ALL)
mat.view()


exit(1)

nx, ny = (1000, 1000)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)

func_x = jax.vmap(lambda x,y: jnp.sin(jnp.pi*(x))*jnp.sin(jnp.pi*(y)),in_axes=(0,0))
func_b = jax.vmap(lambda x,y: -2*jnp.pi**2 * jnp.sin(jnp.pi*(x))*jnp.sin(jnp.pi*(y)),in_axes=(0,0))

xv = xv.flatten()
yv = yv.flatten()

vals = func_b(xv,yv)

lapMatCSR = LaplacianNd((nx,ny),boundary_conditions='dirichlet').tosparse()

data = jnp.asarray(lapMatCSR.data,dtype=jnp.float32)
indices = jnp.asarray(lapMatCSR.indices,dtype=jnp.int32)
indptr = jnp.asarray(lapMatCSR.indptr,dtype=jnp.int32)

"""
results_jax = sparse.linalg.spsolve(data,indices,indptr,vals)

plt.scatter(xv,yv,c=results_jax)
plt.show()

"""

lapMatCOO = lapMatCSR.tocoo()
data = jnp.asarray(lapMatCOO.data,dtype=jnp.float32)
row = jnp.asarray(lapMatCOO.row,dtype=jnp.int32)
col = jnp.asarray(lapMatCOO.col,dtype=jnp.int32)


#results = sparse.linalg.spsolve(data,indices,intptr,vals) fails, not enough memory

comm = PETSc.COMM_SELF
mat = PETSc.Mat().create(comm=comm)
mat.setType(PETSc.Mat.Type.AIJCUSPARSE)
mat.setSizes([nx**2, ny**2])
mat.setPreallocationCOO(row,col)
mat.setValuesCOO(data)

mat.assemblyBegin()
mat.assemblyEnd()

vec = PETSc.Vec().createWithArray(vals)
vec.setType(PETSc.Vec.Type.SEQCUDA)

x = vec.duplicate()


ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setOperators(mat)
ksp.setType('qmrcgs')
ksp.solve(vec,x)

results_petsc = x.getArray()

plt.scatter(xv,yv,c=results_petsc)
plt.show()

vals_x = func_x(xv,yv)

plt.scatter(xv,yv,c=vals_x)
plt.show()
