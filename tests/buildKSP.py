import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.buffer_callback import buffer_callback
import ctypes as ct
import cupy as cp
from flax import struct

import petsc4py
from jax._src.lib import xla_client
from petsc4py import PETSc

_OBJECT_STORE = {}
_NEXT_ID = 0

def __store_object(obj):
    global _NEXT_ID
    uid = _NEXT_ID
    _OBJECT_STORE[uid] = obj
    _NEXT_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type


def __retrieve_object(uid):
    return _OBJECT_STORE[int(uid)]

@struct.dataclass
class __CupyCtx:
    handle: jnp.ndarray


def __petsc_init_impl(ctx, out,jaxMatShape,jaxMatVals,jaxMatRows,jaxMatCols):
    
    jacMatShape = cp.from_dlpack(jaxMatShape,copy=False)
    jacMatVals  = cp.from_dlpack(jaxMatVals,copy=False)
    jacMatRows  = jnp.asarray(jaxMatRows,dtype=jnp.int32)
    jacMatCols  = jnp.asarray(jaxMatCols,dtype=jnp.int32)


    mat = PETSc.Mat().create(PETSc.COMM_WORLD)
    mat.setSizes(jacMatShape)

    mat.setType(PETSc.Mat.Type.AIJCUSPARSE)
    mat.setPreallocationCOO(jacMatRows,jacMatCols)

    lib = ct.CDLL(PETSc.__file__)  # load the PETSc module as a shared library to gain access to the PETSc shared library symbols.
    MatSetValuesCOO = lib.MatSetValuesCOO  # This is the symbol you want to call
    MatSetValuesCOO.restype = ct.c_int  # PetscErrorCode is just a C `int` in terms of ABI.
    MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int] # [Mat, PetscScalar*, InsertMode], I'm using void* instead of PetscScalar* for simplicy, could use `ct.POINTER(ct.c_{float|double})` instead.
    mat_ptr = ct.c_void_p(mat.handle)  # the low level pointer of the mat object
    coo_ptr = ct.c_void_p(jacMatVals.data.ptr)  # the pointer to GPU memory


    MatSetValuesCOO(mat_ptr, coo_ptr, PETSc.InsertMode.INSERT_ALL)
    
    #matdupe = mat.duplicate(copy=True)

    ksp = PETSc.KSP().create()
    ksp.setOperators(mat)
    ksp.setType("lgmres")            # Figure out a better way of
    ksp.setConvergenceHistory()      # setting this option instead
    ksp.getPC().setType("jacobi")    # of hardcoding it like this

    cp.asarray(out)[...] = __store_object(ksp)

@jax.jit   #this needs a jvp and vjp that don't do anything since this is construction
def __petsc_init(jacMatShape,jacMatVals,jacMatRows,jacMatCols) -> __CupyCtx:
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    #handle = jax.pure_callback(__petsc_init_impl, result_info, coo_to_csr(A))
    handle = buffer_callback(__petsc_init_impl, result_info, vmap_method="sequential")(jacMatShape,jacMatVals,jacMatRows,jacMatCols)
    return __CupyCtx(handle=handle)


#this can only be done after we use automatic differentiation because if we try while there's nothing in the state it'll break
def __petsc_cleanup(handle):

    ksp = __retrieve_object(cp.asarray(handle))

    A,P = ksp.getOperators()
    A.destroy()
    P.destroy()
    ksp.getPC().destroy()
    ksp.destroy() #quick and dirty memory management
