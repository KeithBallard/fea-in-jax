from petsc4py import PETSc
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
import jax.dlpack as jdl
import cupy as cp

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


# TODO doesn't work yet, fix

"""
primals, tangents = jax.jvp(jax.numpy.sin, (0.1,), y)

class JaxJvp:

    def __init__(self):
        self.N = (N, N)

    def create(self, A):
        mat_size = A.getSize()

    def mult(self, A, x: , y):
        "y <- A * x"
        N, F = self.N, self.F
        # get 3D arrays from vectos
        xx = x[...].reshape(N, order='f')
        yy = y[...].reshape(N, order='f')
        # call Fortran subroutine
        del2apply(F, xx, yy)

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)

    def getDiagonal(self, A, D):
        "D[i] <- A[i,i]"
        D[...] = 6.0


A = PETSc.Mat().create()
A.setSizes([n**3, n**3])
A.setType('python')
shell = Del2Mat() # shell context
A.setPythonContext(shell)
A.setUp()
"""