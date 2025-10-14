import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.extend

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def __solve_cpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a CPU backend.
    Returns the solution, x.
    """
    # Get the permutation that sorts the matrix entries
    perm = jnp.lexsort((A.col, A.row))

    # Apply the permutation
    sorted_data = np.array(A.data[perm])
    sorted_cols = np.array(A.col[perm])
    sorted_rows = A.row[perm]  # We need this for the next step

    # Count the number of non-zero elements in each row.
    # The 'length' argument is crucial to ensure the output array has size num_rows,
    # even if the last rows are empty.
    num_rows, _ = A.shape
    nnz_per_row = jnp.bincount(sorted_rows, length=num_rows)

    # Build the index pointer array (indptr) from the counts.
    # This is a cumulative sum of the non-zero counts per row.
    # The first element of indptr is always 0.
    indptr = np.array(jnp.concatenate([jnp.array([0]), jnp.cumsum(nnz_per_row)]))

    A_csr = scipy.sparse.csr_matrix(
        (sorted_data, sorted_cols, indptr), shape=(A.shape[0], A.shape[1])
    )

    return scipy.sparse.linalg.spsolve(A_csr, b)


#@jax.jit
def __solve_gpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a GPU backend.
    Returns the solution, x.
    """
    # Get the permutation that sorts the matrix entries
    perm = jnp.lexsort((A.col, A.row))

    # Apply the permutation
    sorted_data = A.data[perm]
    sorted_cols = A.col[perm]
    sorted_rows = A.row[perm]  # We need this for the next step

    # Count the number of non-zero elements in each row.
    # The 'length' argument is crucial to ensure the output array has size num_rows,
    # even if the last rows are empty.
    num_rows, _ = A.shape
    nnz_per_row = jnp.bincount(sorted_rows, length=num_rows)

    # Build the index pointer array (indptr) from the counts.
    # This is a cumulative sum of the non-zero counts per row.
    # The first element of indptr is always 0.
    indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(nnz_per_row)])

    return jsparse.linalg.spsolve(sorted_data, sorted_cols.astype(jnp.int32), indptr.astype(jnp.int32), b)


def solve_sp(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b.
    Returns the solution, x.
    """
    match jax.extend.backend.get_backend().platform:
        case "cpu":
            return __solve_cpu(A, b)
        case "gpu":
            return __solve_gpu(A, b)
    raise Exception(f"Backend {jax.extend.backend.get_backend().platform} unsupported.")

