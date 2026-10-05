import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.buffer_callback import buffer_callback
from jax.interpreters import ad, batching, mlir
import ctypes as ct
import cupy as cp
from flax import struct

import petsc4py
from jax._src.lib import xla_client
from petsc4py import PETSc

_SOLVER_STORE = {}
_PRECOND_STORE = {}
_MATRIX_STORE = {}
_SOLVER_ID = 0        #this maybe needs reworking so we can reuse IDS
_PRECOND_ID = 0
_MATRIX_ID = 0

try:
    from jax.extend import core as jax_core
except ImportError:
    from jax import core as jax_core

try:
    from jax import core as legacy_core
except ImportError:
    legacy_core = jax_core




"""
IMPORTANT!

Buffer callbacks are used because otherwise JAX is unwilling to donate memory. I agree this 
results in some weird looking callstacks and seemingly redundant differentiation rules
but without a lot of this stuff JAX or PETSc will complain and it will either rapidly fill up 
all your memory, fail to differentiate before crashing or if you're very lucky both 

I am going to be cleaning these for legibility and redundancy, but for now be careful
touching these directly. Use/edit the methods defined in JaxCallsPETSc unless you really
like gambling on wasting wall time.

-Alberto
"""



#these are primitives which help us define differentiation
linear_solver_init_p = jax_core.Primitive("v9_keith_linear_solver_init")
linear_solver_cleanup_p = jax_core.Primitive("v9_keith_linear_solver_cleanup")
linear_matrix_init_p = jax_core.Primitive("v9_keith_linear_matrix_init")
linear_matrix_update_p = jax_core.Primitive("v9_keith_linear_matrix_update")
linear_matrix_cleanup_p = jax_core.Primitive("v9_keith_linear_matrix_cleanup")
linear_pc_init_p = jax_core.Primitive("v9_keith_linear_pc_init")
linear_pc_cleanup_p = jax_core.Primitive("v9_keith_linear_pc_cleanup")
linear_ksp_init_p = jax_core.Primitive("v9_keith_linear_ksp_init")


def __handle_to_int(uid):
    try:
        return int(uid)
    except TypeError:
        return int(cp.asarray(uid).item())


#stores linear sovler and possibly metadata
def __store_KSP(obj, owns_operators=False):
    global _SOLVER_ID
    uid = _SOLVER_ID
    _SOLVER_STORE[uid] = (obj, owns_operators)
    _SOLVER_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type


#returns just the linear solver from dictionary
def __retrieve_KSP(uid):
    return _SOLVER_STORE[__handle_to_int(uid)][0]

#returns linear solver from dicctionary WITH METADATA
def __retrieve_KSP_record(uid):
    return _SOLVER_STORE[__handle_to_int(uid)]

#removes the linear solver object from the dictionary
def __CLEAR_KSP(uid):
    _SOLVER_STORE.pop(__handle_to_int(uid))

#stores the preconditioner object in a dictionary
def __store_PC(obj):
    global _PRECOND_ID
    uid = _PRECOND_ID
    _PRECOND_STORE[uid] = obj
    _PRECOND_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type

#function that returns the preconditioner from the dictionary based on the given ID
def __retrieve_PC(uid):
    return _PRECOND_STORE[__handle_to_int(uid)]

#function that removed the preconditioner from the dictionary
def __CLEAR_PC(uid):
    _PRECOND_STORE.pop(__handle_to_int(uid))

#stores a matrix object in a dictionary
def __store_MAT(obj):
    global _MATRIX_ID
    uid = _MATRIX_ID
    _MATRIX_STORE[uid] = obj
    _MATRIX_ID += 1
    return np.int64(uid)

#returns a matrix object from the dictionary
def __retrieve_MAT(uid):
    return _MATRIX_STORE[__handle_to_int(uid)]

#clears a matrix object from the dictionary
def __CLEAR_MAT(uid):
    _MATRIX_STORE.pop(__handle_to_int(uid))


#Pointer for dictionary. Must be this way because otherwise good luck passing it around without staging on CPU
@struct.dataclass
class __CupyCtx:
    handle: jnp.ndarray


#This calls the C (NOT THE PYTHON ONE!) PETSc setValues to minimze overhead. This assumes the memory has been already been allocated
def __mat_set_values_coo(mat, mat_vals):
    lib = ct.CDLL(PETSc.__file__)
    MatSetValuesCOO = lib.MatSetValuesCOO
    MatSetValuesCOO.restype = ct.c_int
    MatSetValuesCOO.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
    err = MatSetValuesCOO(
        ct.c_void_p(mat.handle),
        ct.c_void_p(mat_vals.data.ptr),
        PETSc.InsertMode.INSERT_VALUES,
    )
    if err:
        raise RuntimeError(f"MatSetValuesCOO failed with PETSc error code {err}")

def _zero_from_value(value):
    if hasattr(ad.Zero, "from_primal_value"):
        return ad.Zero.from_primal_value(value)
    aval = jax.typeof(value) if hasattr(jax, "typeof") else legacy_core.get_aval(value)
    return ad.Zero(aval)

#I have no idea what this does
def linearSolverInit(jac, res=None, diag=None, x0=None, constructionOptions=None):
    x0_arg = jnp.array(0, dtype=jnp.int32) if x0 is None else x0
    handle = linear_solver_init_p.bind(
        x0_arg,
        jac=jac,
        res=res,
        diag=diag,
        x0_is_none=x0 is None,
        constructionOptions=constructionOptions,
    )
    return __CupyCtx(handle=handle)

#This is a weird one since it's a _linear function but generally
#this is the implementation of the function that builds the KSP direct from a JAX callable without an intermediate MAT object
def _linear_solver_init_impl(x0, *, jac, res, diag, x0_is_none, constructionOptions):
    x0_value = None if x0_is_none else x0
    J = jac(x0_value)
    if res is not None:
        res(x0_value)
    if diag is not None:
        diag(x0_value)

    return __petsc_KSP_init(J[0], J[1], J[2], J[3], constructionOptions).handle #so this is actually PETSc making a copy and consequently we must assure that the 2 versions of A match

#I have no idea what this does
def _linear_solver_init_abstract_eval(x0_aval, *, jac, res, diag, x0_is_none, constructionOptions):
    del x0_aval, jac, res, diag, x0_is_none, constructionOptions
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array((), jnp.int64)

#I have no idea what this does
def _linear_solver_init_jvp(primals, tangents, *, jac, res, diag, x0_is_none, constructionOptions):
    del tangents
    (x0,) = primals
    handle = linear_solver_init_p.bind(
        x0,
        jac=jac,
        res=res,
        diag=diag,
        x0_is_none=x0_is_none,
        constructionOptions=constructionOptions,
    )
    return handle, _zero_from_value(handle)

#I have no idea what this does
def _linear_solver_init_transpose(ct, x0, *, jac, res, diag, x0_is_none, constructionOptions):
    del ct, x0, jac, res, diag, x0_is_none, constructionOptions
    return (None,)

#This theoretically allows for batched initiation, but given we have like 10 different versions of intiation it's on hold
def _linear_solver_init_batch(args, batch_dims, *, jac, res, diag, x0_is_none, constructionOptions):
    del batch_dims
    raise NotImplementedError("Batching over PETSc solver initialization is not supported")

#This is the called function that from a JAX matrix builds a PETSc MAT object and puts it into a dictionary
def __petsc_MAT_init_impl(ctx, out, jaxMatShape, jaxMatVals, jaxMatRows, jaxMatCols, passedConstructionOptions):
    jacMatShape = cp.from_dlpack(jaxMatShape, copy=False)
    jacMatVals = cp.from_dlpack(jaxMatVals, copy=False)
    jacMatRows = jnp.asarray(jaxMatRows, dtype=jnp.int32)
    jacMatCols = jnp.asarray(jaxMatCols, dtype=jnp.int32)

    constructionOptions = passedConstructionOptions or [PETSc.Mat.Type.AIJCUSPARSE]

    mat = PETSc.Mat().create(PETSc.COMM_WORLD)
    mat.setSizes(jacMatShape)
    mat.setType(constructionOptions[0])
    mat.setPreallocationCOO(jacMatRows, jacMatCols)
    __mat_set_values_coo(mat, jacMatVals)

    cp.asarray(out)[...] = __store_MAT(mat)

#this is the calling function for turning a JAX matrix into a PETSc MAT. This is the one that is built with all JAX understood functions
def __petsc_MAT_init(jacMatShape, jacMatVals, jacMatRows, jacMatCols, constructionOptions=None) -> __CupyCtx:
    result_info = jax.ShapeDtypeStruct((), jnp.int64)

    def callback(ctx, out, shape, vals, rows, cols):
        return __petsc_MAT_init_impl(ctx, out, shape, vals, rows, cols, constructionOptions)

    handle = buffer_callback(callback, result_info, vmap_method="sequential")(
        jacMatShape,
        jacMatVals,
        jacMatRows,
        jacMatCols,
    )
    return __CupyCtx(handle=handle)

#This calls MAT creation from a COOData object instead of a callable
def linearMatrixInitFromCOOData(shape, vals, rows, cols, constructionOptions=None):
    """Private level-4 direct Mat construction from explicit COO arrays."""
    return __petsc_MAT_init(shape, vals, rows, cols, constructionOptions)

#This called method updates the contents of a MAT object without changing the sparsity pattern
def __petsc_MAT_update_impl(ctx, out, handle, jaxMatVals):
    mat = __retrieve_MAT(cp.asarray(handle))
    jacMatVals = cp.from_dlpack(jaxMatVals, copy=False)
    __mat_set_values_coo(mat, jacMatVals)
    cp.asarray(out)[...] = cp.asarray(handle)

#This is the function that calls the function that updates a MAT object. Made of JAX friendly functions and math
def __petsc_MAT_update(handle, jacMatVals) -> __CupyCtx:
    raw_handle = getattr(handle, "handle", handle)
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    out_handle = buffer_callback(__petsc_MAT_update_impl, result_info, vmap_method="sequential")(
        raw_handle,
        jacMatVals, #this is a weird callback shape, why is this way?
    )
    return __CupyCtx(handle=out_handle)

#This function destroys the PETSc MAT associated with the handle
def __petsc_MAT_cleanup(handle):
    mat = __retrieve_MAT(cp.asarray(handle))
    mat.destroy()

#This called function both destroys the MAT object and the dictionary reference
def __petsc_MAT_cleanup_impl(ctx, out, handle):
    __petsc_MAT_cleanup(handle)
    __CLEAR_MAT(handle)
    cp.asarray(out)[...] = cp.asarray(handle)

#This function calls both MAT cleanup routines in a way that JAX understands
def __petsc_MAT_cleanup_callback(handle):
    raw_handle = getattr(handle, "handle", handle)
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    out_handle = buffer_callback(__petsc_MAT_cleanup_impl, result_info, vmap_method="sequential")(raw_handle)
    return __CupyCtx(handle=out_handle)

#No idea what this does
def linearMatrixInit(jac, x0=None, constructionOptions=None):
    x0_arg = jnp.array(0, dtype=jnp.int32) if x0 is None else x0
    handle = linear_matrix_init_p.bind(
        x0_arg,
        jac=jac,
        x0_is_none=x0 is None,
        constructionOptions=constructionOptions,
    )
    return __CupyCtx(handle=handle)

#no idea what this does
def linearMatrixUpdate(matrixHandle, vals):
    raw_handle = getattr(matrixHandle, "handle", matrixHandle)
    handle = linear_matrix_update_p.bind(raw_handle, vals)
    return __CupyCtx(handle=handle)

#no idea that this does
def linearMatrixCleanup(matrixHandle):
    raw_handle = getattr(matrixHandle, "handle", matrixHandle)
    linear_matrix_cleanup_p.bind(raw_handle)
    return matrixHandle

#This called function builds a PC out of a MAT object
def __petsc_PC_init_impl(ctx, out, matrixHandle, passedConstructionOptions):
    mat = __retrieve_MAT(cp.asarray(matrixHandle))

    constructionOptions = passedConstructionOptions or ["jacobi"]

    pc = PETSc.PC().create(PETSc.COMM_WORLD)
    pc.setType(constructionOptions[0])
    pc.setOperators(mat)
    pc.setUp()

    cp.asarray(out)[...] = __store_PC(pc)

#This function calls the function which builds a PC from a MAT objects. This is made of JAX friendly commands
def __petsc_PC_init(matrixHandle, constructionOptions=None) -> __CupyCtx:
    raw_handle = getattr(matrixHandle, "handle", matrixHandle)
    result_info = jax.ShapeDtypeStruct((), jnp.int64)

    def callback(ctx, out, handle):
        return __petsc_PC_init_impl(ctx, out, handle, constructionOptions)

    handle = buffer_callback(callback, result_info, vmap_method="sequential")(
        raw_handle,
    )
    return __CupyCtx(handle=handle)


#Function that destroys the PETSc preconditioner object
def __petsc_PC_cleanup(handle):
    pc = __retrieve_PC(cp.asarray(handle))
    pc.destroy()


#Implementation of the function that cleans up both (PETSc and dictionary) instances of the preconditioner
def __petsc_PC_cleanup_impl(ctx, out, handle):
    __petsc_PC_cleanup(handle)
    __CLEAR_PC(handle)
    cp.asarray(out)[...] = cp.asarray(handle)


#Calling function that JAX is able to see
def __petsc_PC_cleanup_callback(handle):
    raw_handle = getattr(handle, "handle", handle)
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    out_handle = buffer_callback(__petsc_PC_cleanup_impl, result_info, vmap_method="sequential")(raw_handle)
    return __CupyCtx(handle=out_handle)

#No idea what this does
def linearPCInit(matrixHandle, constructionOptions=None):
    raw_handle = getattr(matrixHandle, "handle", matrixHandle)
    handle = linear_pc_init_p.bind(raw_handle, constructionOptions=constructionOptions)
    return __CupyCtx(handle=handle)

#No idea what this does
def linearPCCleanup(pcHandle):
    raw_handle = getattr(pcHandle, "handle", pcHandle)
    linear_pc_cleanup_p.bind(raw_handle)
    return pcHandle

#Okay, unlike the previous one this called function creates a KSP which holds AN ALREADY EXISTING PETSc MAT
#I repeat, unlike the previous one, this one requires a call to the build MAT routines
def __petsc_KSP_init_from_mat_impl(ctx, out, matrixHandle, passedConstructionOptions):
    mat = __retrieve_MAT(cp.asarray(matrixHandle))

    constructionOptions = passedConstructionOptions or ["lgmres"]

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(mat)
    ksp.setType(constructionOptions[0])
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(1e-14, 1e-10, 100, 100000)
    ksp.setConvergenceHistory(3)

    cp.asarray(out)[...] = __store_KSP(ksp, owns_operators=False)

#This is the calling function that uses a constructed MAT to make a KSP
def __petsc_KSP_init_from_mat(matrixHandle, constructionOptions=None) -> __CupyCtx:
    raw_handle = getattr(matrixHandle, "handle", matrixHandle)
    result_info = jax.ShapeDtypeStruct((), jnp.int64)

    def callback(ctx, out, handle):
        return __petsc_KSP_init_from_mat_impl(ctx, out, handle, constructionOptions)

    handle = buffer_callback(callback, result_info, vmap_method="sequential")(
        raw_handle,
    )
    return __CupyCtx(handle=handle)

#No idea what this does
def linearKSPInit(matrixHandle, constructionOptions=None):
    raw_handle = getattr(matrixHandle, "handle", matrixHandle)
    handle = linear_ksp_init_p.bind(raw_handle, constructionOptions=constructionOptions)
    return __CupyCtx(handle=handle)



#This creates a PETSc MAT which is used to create a KSP from JAX vectors. Think of it as a simplification of
#memory management. To be used with broader cleanup calls
def __petsc_KSP_init_impl(ctx, out,jaxMatShape,jaxMatVals,jaxMatRows,jaxMatCols,passedConstructionOptions):
    
    jacMatShape = cp.from_dlpack(jaxMatShape,copy=False)
    jacMatVals  = cp.from_dlpack(jaxMatVals,copy=False)
    jacMatRows  = jnp.asarray(jaxMatRows,dtype=jnp.int32)
    jacMatCols  = jnp.asarray(jaxMatCols,dtype=jnp.int32)

    constructionOptions = passedConstructionOptions or [PETSc.Mat.Type.AIJCUSPARSE, "lgmres", "jacobi"]

    mat = PETSc.Mat().create(PETSc.COMM_WORLD)
    mat.setSizes(jacMatShape)

    mat.setType(constructionOptions[0])#PETSc.Mat.Type.AIJCUSPARSE
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
    ksp.setType(constructionOptions[1])#"lgmres")            # Figure out a better way of
    ksp.setConvergenceHistory()      # setting this option instead
    ksp.getPC().setType(constructionOptions[2])#"jacobi")    # of hardcoding it like this



    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                      #rtol, atol, dtol, max_it
    ksp.setTolerances(1e-14,1e-10, 100, 100000) #careful with these, petsc is quite a bit more rigorusly demanding than scipy 
    n = 3 #constructionOptions
    ksp.setConvergenceHistory(n)

    cp.asarray(out)[...] = __store_KSP(ksp, owns_operators=True)


#this is what JAX calls since it contains only JAX understood maths. This creates a KSP (and only a KSP, no MAT) based on the JAX matrix
def __petsc_KSP_init(jacMatShape,jacMatVals,jacMatRows,jacMatCols,constructionOptions) -> __CupyCtx:
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    #handle = jax.pure_callback(__petsc_init_impl, result_info, coo_to_csr(A))

    def callback(ctx, out, shape, vals, rows, cols):
        return __petsc_KSP_init_impl(ctx, out, shape, vals, rows, cols, constructionOptions)

    handle = buffer_callback(callback, result_info, vmap_method="sequential")(jacMatShape,jacMatVals,jacMatRows,jacMatCols)
    return __CupyCtx(handle=handle)
    
#This just dumps everything associated with a KSP including the KSP
def __petsc_cleanup(handle):

    ksp, owns_operators = __retrieve_KSP_record(cp.asarray(handle))

    if owns_operators:
        A, P = ksp.getOperators()
        seen = set()
        for obj in (A, P, ksp.getPC()):
            obj_handle = getattr(obj, "handle", None)
            if obj_handle not in seen:
                obj.destroy()
                seen.add(obj_handle)

    ksp.destroy() #quick and dirty memory management

#Removed the KSP object from the dictionary
def __dictionary_cleanup(handle):
    __CLEAR_KSP(handle)

#implementation that both destroys the KSP object (and its contents) and the dictionary entry
def __petsc_cleanup_impl(ctx, out, handle):
    __petsc_cleanup(handle)
    __dictionary_cleanup(handle)
    cp.asarray(out)[...] = cp.asarray(handle)

#buffer callback to the cleanup function so JAX understands it
def __petsc_cleanup_callback(handle):
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    return buffer_callback(__petsc_cleanup_impl, result_info, vmap_method="sequential")(handle)

#No idea what this is
def linearSolverCleanup(handle):
    raw_handle = getattr(handle, "handle", handle)
    linear_solver_cleanup_p.bind(raw_handle)
    return handle

#No idea what this is
def _linear_solver_cleanup_impl(handle):
    __petsc_cleanup(handle)
    __dictionary_cleanup(handle)
    return handle

#No idea what this is
def _linear_solver_cleanup_abstract_eval(handle_aval):
    return handle_aval

#No idea what this is
def _linear_solver_init_lowering(ctx, x0, *, jac, res, diag, x0_is_none, constructionOptions):
    def lowered_init(x0_value):
        primal_x0 = None if x0_is_none else x0_value
        J = jac(primal_x0)
        if res is not None:
            res(primal_x0)
        if diag is not None:
            diag(primal_x0)
        return __petsc_KSP_init(J[0], J[1], J[2], J[3], constructionOptions).handle

    return mlir.lower_fun(lowered_init, multiple_results=False)(ctx, x0)

#No idea what this is
def _linear_solver_cleanup_jvp(primals, tangents):
    del tangents
    (handle,) = primals
    out = linear_solver_cleanup_p.bind(handle)
    return out, _zero_from_value(out)

#No idea what this is
def _linear_solver_cleanup_transpose(ct, handle):
    del ct, handle
    return (None,)

#No idea what this is
def _linear_solver_cleanup_batch(args, batch_dims):
    del args, batch_dims
    raise NotImplementedError("Batching over PETSc solver cleanup is not supported")

#No idea what this is
def _linear_solver_cleanup_lowering(ctx, handle):
    return mlir.lower_fun(__petsc_cleanup_callback, multiple_results=False)(ctx, handle)

#No idea what this is
def _linear_matrix_init_impl(x0, *, jac, x0_is_none, constructionOptions):
    x0_value = None if x0_is_none else x0
    J = jac(x0_value)
    return __petsc_MAT_init(J[0], J[1], J[2], J[3], constructionOptions).handle

#No idea what this is
def _linear_matrix_init_abstract_eval(x0_aval, *, jac, x0_is_none, constructionOptions):
    del x0_aval, jac, x0_is_none, constructionOptions
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array((), jnp.int64)

#No idea what this is
def _linear_matrix_init_jvp(primals, tangents, *, jac, x0_is_none, constructionOptions):
    del tangents
    (x0,) = primals
    handle = linear_matrix_init_p.bind(
        x0,
        jac=jac,
        x0_is_none=x0_is_none,
        constructionOptions=constructionOptions,
    )
    return handle, _zero_from_value(handle)

#No idea what this is
def _linear_matrix_init_transpose(ct, x0, *, jac, x0_is_none, constructionOptions):
    del ct, x0, jac, x0_is_none, constructionOptions
    return (None,)

#No idea what this is
def _linear_matrix_init_batch(args, batch_dims, *, jac, x0_is_none, constructionOptions):
    del args, batch_dims, jac, x0_is_none, constructionOptions
    raise NotImplementedError("Batching over PETSc matrix initialization is not supported")

#No idea what this is
def _linear_matrix_init_lowering(ctx, x0, *, jac, x0_is_none, constructionOptions):
    def lowered_init(x0_value):
        primal_x0 = None if x0_is_none else x0_value
        J = jac(primal_x0)
        return __petsc_MAT_init(J[0], J[1], J[2], J[3], constructionOptions).handle

    return mlir.lower_fun(lowered_init, multiple_results=False)(ctx, x0)

#No idea what this is
def _linear_matrix_update_impl(handle, vals):
    return __petsc_MAT_update(handle, vals).handle

#No idea what this is
def _linear_matrix_update_abstract_eval(handle_aval, vals_aval):
    del vals_aval
    return handle_aval

#No idea what this is
def _linear_matrix_update_jvp(primals, tangents):
    del tangents
    handle, vals = primals
    out = linear_matrix_update_p.bind(handle, vals)
    return out, _zero_from_value(out)

#No idea what this is
def _linear_matrix_update_transpose(ct, handle, vals):
    del ct, handle
    vals_bar = ad.Zero(vals.aval) if isinstance(vals, ad.UndefinedPrimal) else None
    return None, vals_bar

#No idea what this is
def _linear_matrix_update_batch(args, batch_dims):
    del args, batch_dims
    raise NotImplementedError("Batching over PETSc matrix updates is not supported")

#No idea what this is
def _linear_matrix_update_lowering(ctx, handle, vals):
    def lowered_update(handle_value, vals_value):
        return __petsc_MAT_update(handle_value, vals_value).handle

    return mlir.lower_fun(lowered_update, multiple_results=False)(ctx, handle, vals)

#No idea what this is
def _linear_matrix_cleanup_impl(handle):
    return __petsc_MAT_cleanup_callback(handle).handle

#No idea what this is
def _linear_matrix_cleanup_abstract_eval(handle_aval):
    return handle_aval

#No idea what this is
def _linear_matrix_cleanup_jvp(primals, tangents):
    del tangents
    (handle,) = primals
    out = linear_matrix_cleanup_p.bind(handle)
    return out, _zero_from_value(out)

#No idea what this is
def _linear_matrix_cleanup_transpose(ct, handle):
    del ct, handle
    return (None,)

#No idea what this is
def _linear_matrix_cleanup_batch(args, batch_dims):
    del args, batch_dims
    raise NotImplementedError("Batching over PETSc matrix cleanup is not supported")

#No idea what this is
def _linear_matrix_cleanup_lowering(ctx, handle):
    return mlir.lower_fun(lambda h: __petsc_MAT_cleanup_callback(h).handle, multiple_results=False)(ctx, handle)

#No idea what this is
def _linear_pc_init_impl(matrix_handle, *, constructionOptions):
    return __petsc_PC_init(matrix_handle, constructionOptions).handle

#No idea what this is
def _linear_pc_init_abstract_eval(matrix_handle_aval, *, constructionOptions):
    del matrix_handle_aval, constructionOptions
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array((), jnp.int64)

#No idea what this is
def _linear_pc_init_jvp(primals, tangents, *, constructionOptions):
    del tangents
    (matrix_handle,) = primals
    handle = linear_pc_init_p.bind(matrix_handle, constructionOptions=constructionOptions)
    return handle, _zero_from_value(handle)

#No idea what this is
def _linear_pc_init_transpose(ct, matrix_handle, *, constructionOptions):
    del ct, matrix_handle, constructionOptions
    return (None,)

#No idea what this is
def _linear_pc_init_batch(args, batch_dims, *, constructionOptions):
    del args, batch_dims, constructionOptions
    raise NotImplementedError("Batching over PETSc PC initialization is not supported")

#No idea what this is
def _linear_pc_init_lowering(ctx, matrix_handle, *, constructionOptions):
    return mlir.lower_fun(lambda h: __petsc_PC_init(h, constructionOptions).handle, multiple_results=False)(ctx, matrix_handle)

#No idea what this is
def _linear_pc_cleanup_impl(handle):
    return __petsc_PC_cleanup_callback(handle).handle

#No idea what this is
def _linear_pc_cleanup_abstract_eval(handle_aval):
    return handle_aval

#No idea what this is
def _linear_pc_cleanup_jvp(primals, tangents):
    del tangents
    (handle,) = primals
    out = linear_pc_cleanup_p.bind(handle)
    return out, _zero_from_value(out)

#No idea what this is
def _linear_pc_cleanup_transpose(ct, handle):
    del ct, handle
    return (None,)

#No idea what this is
def _linear_pc_cleanup_batch(args, batch_dims):
    del args, batch_dims
    raise NotImplementedError("Batching over PETSc PC cleanup is not supported")

#No idea what this is
def _linear_pc_cleanup_lowering(ctx, handle):
    return mlir.lower_fun(lambda h: __petsc_PC_cleanup_callback(h).handle, multiple_results=False)(ctx, handle)

#No idea what this is
def _linear_ksp_init_impl(matrix_handle, *, constructionOptions):
    return __petsc_KSP_init_from_mat(matrix_handle, constructionOptions).handle

#No idea what this is
def _linear_ksp_init_abstract_eval(matrix_handle_aval, *, constructionOptions):
    del matrix_handle_aval, constructionOptions
    shaped_array = getattr(jax_core, "ShapedArray", legacy_core.ShapedArray)
    return shaped_array((), jnp.int64)

#No idea what this is
def _linear_ksp_init_jvp(primals, tangents, *, constructionOptions):
    del tangents
    (matrix_handle,) = primals
    handle = linear_ksp_init_p.bind(matrix_handle, constructionOptions=constructionOptions)
    return handle, _zero_from_value(handle)

#No idea what this is
def _linear_ksp_init_transpose(ct, matrix_handle, *, constructionOptions):
    del ct, matrix_handle, constructionOptions
    return (None,)

#No idea what this is
def _linear_ksp_init_batch(args, batch_dims, *, constructionOptions):
    del args, batch_dims, constructionOptions
    raise NotImplementedError("Batching over PETSc KSP initialization is not supported")

#No idea what this is
def _linear_ksp_init_lowering(ctx, matrix_handle, *, constructionOptions):
    return mlir.lower_fun(lambda h: __petsc_KSP_init_from_mat(h, constructionOptions).handle, multiple_results=False)(ctx, matrix_handle)


linear_solver_init_p.def_impl(_linear_solver_init_impl)
linear_solver_init_p.def_abstract_eval(_linear_solver_init_abstract_eval)
mlir.register_lowering(linear_solver_init_p, _linear_solver_init_lowering)
ad.primitive_jvps[linear_solver_init_p] = _linear_solver_init_jvp
ad.primitive_transposes[linear_solver_init_p] = _linear_solver_init_transpose
batching.primitive_batchers[linear_solver_init_p] = _linear_solver_init_batch

linear_solver_cleanup_p.def_impl(_linear_solver_cleanup_impl)
linear_solver_cleanup_p.def_abstract_eval(_linear_solver_cleanup_abstract_eval)
mlir.register_lowering(linear_solver_cleanup_p, _linear_solver_cleanup_lowering)
ad.primitive_jvps[linear_solver_cleanup_p] = _linear_solver_cleanup_jvp
ad.primitive_transposes[linear_solver_cleanup_p] = _linear_solver_cleanup_transpose
batching.primitive_batchers[linear_solver_cleanup_p] = _linear_solver_cleanup_batch

linear_matrix_init_p.def_impl(_linear_matrix_init_impl)
linear_matrix_init_p.def_abstract_eval(_linear_matrix_init_abstract_eval)
mlir.register_lowering(linear_matrix_init_p, _linear_matrix_init_lowering)
ad.primitive_jvps[linear_matrix_init_p] = _linear_matrix_init_jvp
ad.primitive_transposes[linear_matrix_init_p] = _linear_matrix_init_transpose
batching.primitive_batchers[linear_matrix_init_p] = _linear_matrix_init_batch

linear_matrix_update_p.def_impl(_linear_matrix_update_impl)
linear_matrix_update_p.def_abstract_eval(_linear_matrix_update_abstract_eval)
mlir.register_lowering(linear_matrix_update_p, _linear_matrix_update_lowering)
ad.primitive_jvps[linear_matrix_update_p] = _linear_matrix_update_jvp
ad.primitive_transposes[linear_matrix_update_p] = _linear_matrix_update_transpose
batching.primitive_batchers[linear_matrix_update_p] = _linear_matrix_update_batch

linear_matrix_cleanup_p.def_impl(_linear_matrix_cleanup_impl)
linear_matrix_cleanup_p.def_abstract_eval(_linear_matrix_cleanup_abstract_eval)
mlir.register_lowering(linear_matrix_cleanup_p, _linear_matrix_cleanup_lowering)
ad.primitive_jvps[linear_matrix_cleanup_p] = _linear_matrix_cleanup_jvp
ad.primitive_transposes[linear_matrix_cleanup_p] = _linear_matrix_cleanup_transpose
batching.primitive_batchers[linear_matrix_cleanup_p] = _linear_matrix_cleanup_batch

linear_pc_init_p.def_impl(_linear_pc_init_impl)
linear_pc_init_p.def_abstract_eval(_linear_pc_init_abstract_eval)
mlir.register_lowering(linear_pc_init_p, _linear_pc_init_lowering)
ad.primitive_jvps[linear_pc_init_p] = _linear_pc_init_jvp
ad.primitive_transposes[linear_pc_init_p] = _linear_pc_init_transpose
batching.primitive_batchers[linear_pc_init_p] = _linear_pc_init_batch

linear_pc_cleanup_p.def_impl(_linear_pc_cleanup_impl)
linear_pc_cleanup_p.def_abstract_eval(_linear_pc_cleanup_abstract_eval)
mlir.register_lowering(linear_pc_cleanup_p, _linear_pc_cleanup_lowering)
ad.primitive_jvps[linear_pc_cleanup_p] = _linear_pc_cleanup_jvp
ad.primitive_transposes[linear_pc_cleanup_p] = _linear_pc_cleanup_transpose
batching.primitive_batchers[linear_pc_cleanup_p] = _linear_pc_cleanup_batch

linear_ksp_init_p.def_impl(_linear_ksp_init_impl)
linear_ksp_init_p.def_abstract_eval(_linear_ksp_init_abstract_eval)
mlir.register_lowering(linear_ksp_init_p, _linear_ksp_init_lowering)
ad.primitive_jvps[linear_ksp_init_p] = _linear_ksp_init_jvp
ad.primitive_transposes[linear_ksp_init_p] = _linear_ksp_init_transpose
batching.primitive_batchers[linear_ksp_init_p] = _linear_ksp_init_batch

