import jax
from petsc4py import PETSc
from jax import random
import jax.numpy as jnp
from jax.experimental import sparse
import ctypes as ct
import cupy as cp
from flax import struct
import numpy as np
from jax.experimental.buffer_callback import buffer_callback

jax.config.update("jax_enable_x64", True)


seed = 1975
key = jax.random.key(seed)

test1 = 100000
test2 = 800000

vecSize = test2

@struct.dataclass
class __CupyCtx:
    handle: jnp.ndarray

@struct.dataclass
class __CupyCtx:
    handle: jnp.ndarray
    
    
@jax.jit
def __petsc_solve(ctx: __CupyCtx, b: jnp.ndarray):
    result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return buffer_callback(__petsc_solve_impl_debug, result_info)(ctx.handle, b)

def __petsc_solve_impl_debug(ctx, out, handle: jnp.ndarray, b: jnp.ndarray):
    
    GPUPointerArray = cp.from_dlpack(b,copy=False)
    b_petsc_1 = PETSc.Vec().createWithDLPack(GPUPointerArray, size=b.shape[0])
    

    ksp = __retrieve_object(cp.asarray(handle))
    
    x_petsc = PETSc.Vec().create(PETSc.COMM_SELF)
    x_petsc.setType('cuda')         # true GPU vector
    x_petsc.setSizes(b.shape[0])
    x_petsc.setUp()
    x_petsc.set(0.0)

    n = 3

    ksp.setTolerances(rtol=1e-9,atol=1e-9)
    
    ksp.setConvergenceHistory(n)
    ksp.solve(b_petsc_1,x_petsc)
    

    cudahandle = x_petsc.getCUDAHandle()
    ptr = cudahandle         # raw CUDA pointer from PETSc
    length = x_petsc.getSize()
     
    x_gpu = cp.ndarray((length,), dtype=cp.float64    , memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, length*8, x_petsc), 0))

    print("x First 10 elements inside buffer_callback:", x_gpu[0:10])

        
    convergenceHist = ksp.getConvergenceHistory()

    #print(convergenceHist)

    __store_solution(cp.asarray((x_petsc.getArray())))

    ksp.destroy() #quick and dirty memory management

    cp.asarray(out)[...] = cp.asarray(x_petsc.getArray())
    
@jax.jit
def __petsc_init(A: sparse.COO) -> __CupyCtx:
    result_info = jax.ShapeDtypeStruct((), jnp.int64)
    print("got result shape")
    handle = jax.pure_callback(__petsc_init_impl, result_info, coo_to_csr(A))
    return __CupyCtx(handle=handle)



def __petsc_init_impl(A: sparse.CSR):
    A_petsc = PETSc.Mat()
    A_petsc.create(PETSc.COMM_WORLD)
    A_petsc.setSizes([A.shape[0], A.shape[1]])
    
    A_petsc.createAIJWithArrays(
        size=(A.shape[0], A.shape[1]),
        csr=(
            cp.asarray(A.indptr).get().astype(np.int32),
            cp.asarray(A.indices).get().astype(np.int32),
            cp.asarray(A.data).get(),
        ),
    )
    # NOTE this appears to be moved to CPU for these calls.
    # TODO figure out how to populate A with GPU arrays.
    A_petsc.setType("aijcusparse")


    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType("cg")
    ksp.setConvergenceHistory()
    ksp.getPC().setType("none")




    return __store_object(ksp)


_OBJECT_STORE = {}
_NEXT_ID = 0

_SOLUTION_STORE = {} #THIS IS MEANT TO HOLD A SOLUTION VECTOR FOR PETSC


def __store_object(obj):
    global _NEXT_ID
    uid = _NEXT_ID
    _OBJECT_STORE[uid] = obj
    _NEXT_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type


def __retrieve_object(uid):
    return _OBJECT_STORE[int(uid)]


def __store_solution(obj):
    global _NEXT_ID
    uid = _NEXT_ID
    _SOLUTION_STORE[uid] = obj
    return np.int64(uid)


@jax.jit
def coo_to_csr(A: sparse.COO):
    """
    Convert a COO sparse matrix to a CSR sparse matrix.

    Args:
        sum_duplicates: indicates whether to sum duplicate indices.

    Returns:
        (data, row, col) defining a COO matrix with duplicates summed.

    IMPORTANT NOTE:
        If the resulting CSR will be used with spsolve, make sure to set sum_duplicates to True
        because the CUDA sparse solver will not yield the correct result.
    """
    if not A._rows_sorted:
        # Get the permutation that sorts the matrix entries
        perm = jnp.lexsort((A.col, A.row))

        # Apply the permutation
        data = A.data[perm]
        rows = A.row[perm]
        cols = A.col[perm]
    else:
        data = A.data
        rows = A.row
        cols = A.col

    # Count the number of non-zero elements in each row.
    # The 'length' argument is crucial to ensure the output array has size num_rows,
    # even if the last rows are empty.
    num_rows, _ = A.shape
    nnz_per_row = jnp.bincount(rows, length=num_rows)

    # Build the index pointer array (indptr) from the counts.
    # This is a cumulative sum of the non-zero counts per row.
    # The first element of indptr is always 0.
    indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(nnz_per_row)])

    return sparse.CSR((data, cols, indptr), shape=A.shape)



@jax.jit()
def memSizeTest():
    diag = random.normal(key,shape=(vecSize,))
    rows = jnp.arange(vecSize)
    cols = jnp.arange(vecSize)
    
    vecb = jnp.ones((vecSize,))
    
    sparseMatObj = sparse.COO((diag,rows,cols),shape=(vecSize,vecSize),rows_sorted=True,cols_sorted=True)
    
    exampleOutput = callbackCompute(sparseMatObj,vecb)
    
    jax.debug.print("first ten entries of example: {ten}",ten=exampleOutput[0:10])
    
    


def callbackCompute(mat,vec):
    ctx = __petsc_init(mat)
    delta_x = __petsc_solve(ctx, vec)
    return delta_x


memSizeTest()