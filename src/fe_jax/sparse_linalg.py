import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.extend as jextend

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .utils import debug_print


def coo_sum_duplicates(
    A: jsparse.COO, buffer_result: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns the row-then-column sorted arrays for a new COO matrix after summing
    duplicate indices.

    Args:
        buffer_result: returned arrays will be same length as those in A (allowing JIT).

    Returns:
        (data, row, col) defining a COO matrix with duplicates summed.

    """

    # Credit: https://stackoverflow.com/a/25789764

    # Get the permutation that sorts the matrix entries
    perm = jnp.lexsort((A.col, A.row))

    # Creates an array of (row, col) entries (sorted by row then col using perm)
    sorted_indices = jnp.vstack((A.row[perm], A.col[perm])).T
    # An array of indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(sorted_indices, axis=0)
    # Boolean mask indicating if each (row, col) value is unique
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))

    if not buffer_result:
        # A map from the unique order to the original order
        unique_indices = perm[uniq_mask]
        # A map from the original order to the unique order
        inv_indices = jnp.zeros_like(perm).at[perm].set(jnp.cumsum(uniq_mask) - 1)
        # Effectively sums duplicates and returns the values in the permuated order
        unique_data = jnp.bincount(inv_indices, weights=A.data)
        return (unique_data, A.row[unique_indices], A.col[unique_indices])
    else:
        # Same as above but buffers arrays to allow JIT
        unique_indices = jnp.where(uniq_mask, perm, perm)
        inv_indices = jnp.zeros_like(perm).at[perm].set(jnp.cumsum(uniq_mask) - 1)
        debug_print(inv_indices)
        debug_print(A.data)
        #data = jnp.bincount(inv_indices, weights=A.data, length=perm.shape[0])
        data = A.data
        rows = jnp.zeros_like(A.row).at[unique_indices].set(A.row[unique_indices])
        cols = jnp.zeros_like(A.col).at[unique_indices].set(A.col[unique_indices])
        return (data, rows, cols)


@jax.jit
def coo_to_csr(A: jsparse.COO, sum_duplicates: bool = True):
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

    if sum_duplicates:
        data, rows, cols = coo_sum_duplicates(A, True)
    else:
        # Get the permutation that sorts the matrix entries
        perm = jnp.lexsort((A.col, A.row))

        # Apply the permutation
        data = A.data[perm]
        rows = A.col[perm]
        cols = A.row[perm]

    # Count the number of non-zero elements in each row.
    # The 'length' argument is crucial to ensure the output array has size num_rows,
    # even if the last rows are empty.
    num_rows, _ = A.shape
    nnz_per_row = jnp.bincount(rows, length=num_rows)

    # Build the index pointer array (indptr) from the counts.
    # This is a cumulative sum of the non-zero counts per row.
    # The first element of indptr is always 0.
    indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(nnz_per_row)])

    # Count the number of non-zero elements in each row.
    # The 'length' argument is crucial to ensure the output array has size num_rows,
    # even if the last rows are empty.
    num_rows, _ = A.shape
    nnz_per_row = jnp.bincount(rows, length=num_rows)

    # Build the index pointer array (indptr) from the counts.
    # This is a cumulative sum of the non-zero counts per row.
    # The first element of indptr is always 0.
    indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(nnz_per_row)])

    return jsparse.CSR((data, cols, indptr), shape=A.shape)


def __solve_cpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a CPU backend.
    Returns the solution, x.
    """
    A_jax_csr = coo_to_csr(A)
    A_csr = scipy.sparse.csr_matrix(
        (
            np.array(A_jax_csr.data),
            np.array(A_jax_csr.indices),
            np.array(A_jax_csr.indptr),
        ),
        shape=(A.shape[0], A.shape[1]),
    )
    return scipy.sparse.linalg.spsolve(A_csr, b)


@jax.jit
def __solve_gpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a GPU backend.
    Returns the solution, x.
    """
    A_csr = coo_to_csr(A)
    return jsparse.linalg.spsolve(
        A_csr.data, A_csr.indices.astype(jnp.int32), A_csr.indptr.astype(jnp.int32), b
    )


def spsolve(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b.
    Returns the solution, x.
    """
    match jextend.backend.get_backend().platform:
        case "cpu":
            return __solve_cpu(A, b)
        case "gpu":
            return __solve_gpu(A, b)
    raise Exception(f"Backend {jextend.backend.get_backend().platform} unsupported.")
