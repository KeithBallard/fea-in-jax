import petsc4py
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix

import jax.numpy as jnp
from jax import random
from jax.experimental import sparse

import cupy as cp

import sys
import os.path


testPreallocate1 = False
testPreallocate2 = True
testPreallocate3 = True
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
