import jax
import jax.numpy as jnp
import jax.scipy as jsc

from petsc4py import PETSc

import JaxMat



def jacobianMult(x): #we should maybe turn this into an interface so we guarantee that anything importing this has these methods since they're needed
    ...

def residualMult(x):
    ...

def jacobianDiagMult(x):
    ...
    

def buildPETScJaxMat(shape,jacobianFunc,ResidualFunc=None,DiagFunc=None, dmplex=None, comm=None):

    matClass = JaxMat.JaxMat(dmplex=dmplex,jaxMult=jacobianFunc)
    comm = comm or (dmplex.comm if dmplex is not None else PETSc.COMM_WORLD)

    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes(shape)
    mat.setType(PETSc.Mat.Type.PYTHON)
    mat.setPythonContext(matClass)
    mat.setUp()

    return mat, matClass

def buildPETScKSP(mat,solverType, precondType="none", comm=None):
    
    ksp = PETSc.KSP().create(comm=comm or getattr(mat, "comm", PETSc.COMM_WORLD))
    ksp.setType(solverType)
    ksp.setOperators(mat)
    ksp.getPC().setType(precondType)
    ksp.setFromOptions()

    return ksp

def buildPETScPC(mat,precondType):
    
    pc = PETSc.PC().create()
    pc.setType(precondType)

    pc.setFromOptions()
    pc.setOperators(mat)

    return pc

