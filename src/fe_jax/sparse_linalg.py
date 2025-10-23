import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.extend as jextend

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from functools import partial

from .utils import debug_print


def apply_dirichlet_bcs(
    A: jsparse.COO,
    b: jnp.ndarray,
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
) -> tuple[jsparse.COO, jnp.ndarray]:
    """
    Applies Dirichlet BCs directly to a COO sparse matrix, A, and adjusts the RHS vector, b.
    """

    # Create a mask to filter out rows corresponding to BCs.
    # We want to keep an element if its row index is NOT in bc_indices.
    # Note: jnp.isin returns a boolean array that is True for the Dirichlet BC DOFs, but we want
    # the opposite, hence ~.
    keep_mask = ~jnp.isin(A.row, dirichlet_dofs) & ~jnp.isin(A.col, dirichlet_dofs)
    debug_print(keep_mask)

    # Create the new data entries for 1's on the diagon corresponding to the BCs.
    num_bcs = dirichlet_dofs.shape[0]
    bc_data = jnp.ones(num_bcs, dtype=A.dtype)
    debug_print(bc_data)

    # Create a new sparse matrix by concatenating the old (filtered) and new (BC) entries
    # Note: astype(jnp.int64) is neccessary because jnp.concatenate with (jnp.int64, jnp.uint64)
    # results in a jnp.float64 array for some reason.
    data = jnp.concatenate([jnp.where(keep_mask, A.data, A.data), bc_data])
    rows = jnp.concatenate(
        [jnp.where(keep_mask, A.row, A.row), dirichlet_dofs.astype(jnp.int64)]
    )
    cols = jnp.concatenate(
        [jnp.where(keep_mask, A.col, A.col), dirichlet_dofs.astype(jnp.int64)]
    )
    print(data.shape)
    print(rows.shape)
    print(cols.shape)
    A_modified = jsparse.COO(
        (
            jnp.concatenate([jnp.where(keep_mask, A.data, A.data), bc_data]),
            jnp.concatenate(
                [jnp.where(keep_mask, A.row, A.row), dirichlet_dofs.astype(jnp.int64)]
            ),
            jnp.concatenate(
                [jnp.where(keep_mask, A.col, A.col), dirichlet_dofs.astype(jnp.int64)]
            ),
        ),
        shape=A.shape,
    )
    A_modified = A_modified._sort_indices()
    debug_print(A_modified.data)

    """
    A_modified = jsparse.COO(
        (
            jnp.concatenate([A.data[keep_mask], bc_data]),
            jnp.concatenate([A.row[keep_mask], dirichlet_dofs]),
            jnp.concatenate([A.col[keep_mask], dirichlet_dofs]),
        ),
        shape=A.shape,
    )
    """

    # Update the RHS vector
    tmp = jnp.zeros_like(b)
    tmp = tmp.at[dirichlet_dofs].set(dirichlet_values)
    b_modified = b - A @ tmp
    b_modified = b_modified.at[dirichlet_dofs].set(dirichlet_values)

    return A_modified, b_modified


def coo_arrays_sum_duplicates(A: jsparse.COO) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns the row-then-column sorted arrays for a new COO matrix after summing
    duplicate indices.

    Args:
        A: input matrix for which to sum duplicates.

    Returns:
        (data, row, col) defining a COO matrix with duplicates summed.

    """

    # Credit: https://stackoverflow.com/a/25789764

    # Get the permutation that sorts the matrix entries
    perm = jnp.lexsort((A.col, A.row))
    # Creates an array of (row, col) entries (sorted by row then col using perm)
    sorted_indices = jnp.vstack((A.row[perm], A.col[perm])).T
    # An array of sorted_indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(sorted_indices, axis=0)
    # Boolean mask indicating if each (row, col) value is unique, shape=A.col.shape
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))
    # A map from the unique order to the original order
    unique_indices = perm[uniq_mask]
    # A map from the original order to the unique order
    inv_indices = jnp.zeros_like(perm).at[perm].set(jnp.cumsum(uniq_mask) - 1)
    # Effectively sums duplicates and returns the values in the permuated order
    unique_data = jnp.bincount(inv_indices, weights=A.data)
    return (unique_data, A.row[unique_indices], A.col[unique_indices])


@partial(jax.jit, static_argnames=["result_length"])
def coo_arrays_sum_duplicates_jit(
    A: jsparse.COO, result_length: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns the row-then-column sorted arrays for a new COO matrix after summing
    duplicate indices.

    Args:
        A: input matrix for which to sum duplicates.

        result_length: specified length for resultant arrays (allowing JIT) but should be the
            number of non-zeros after duplicates are combined.

    Returns:
        (data, row, col) defining a COO matrix with duplicates summed.

    """

    # Credit: https://stackoverflow.com/a/25789764

    # Get the permutation that sorts the matrix entries
    perm = jnp.lexsort((A.col, A.row))
    # Creates an array of (row, col) entries (sorted by row then col using perm)
    sorted_indices = jnp.vstack((A.row[perm], A.col[perm])).T
    debug_print(sorted_indices)
    # An array of sorted_indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(sorted_indices, axis=0)
    debug_print(diff)
    # Boolean mask indicating if each (row, col) value is unique, shape=A.col.shape
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))
    debug_print(uniq_mask)
    # A map from the unique order to the original order
    # NOTE: there is a trick here to get the unique indices while also guaranteeing array sizes
    unique_indices = jnp.sort(jnp.where(uniq_mask, perm, jnp.max(perm) + 1))[0:result_length]
    debug_print(unique_indices)
    # A map from the original order to the unique order
    inv_indices = jnp.zeros_like(perm).at[perm].set(jnp.cumsum(uniq_mask) - 1)
    debug_print(inv_indices)
    # Effectively sums duplicates and returns the values in the permuated order
    data = jnp.bincount(inv_indices, weights=A.data, length=result_length)
    rows = A.row[unique_indices]
    cols = A.col[unique_indices]
    debug_print(data)
    debug_print(rows)
    debug_print(cols)
    return (data, rows, cols)


@partial(jax.jit, static_argnames=["result_length"])
def coo_sum_duplicates(A: jsparse.COO, result_length: int = 0) -> jsparse.COO:
    """
    Returns a row-then-column sorted COO matrix after summing duplicate indices.

    Args:
        result_length: specified length for resultant arrays (allowing JIT) but should be the
            number of non-zeros after duplicates are combined. A value of 0 will dynamically
            allocate the arrays but also be incompatible with JIT.

    Returns:
        COO matrix with duplicates summed.

    """
    data, rows, cols = coo_arrays_sum_duplicates_jit(A=A, result_length=result_length)
    return jsparse.COO((data, rows, cols), shape=A.shape, rows_sorted=True)


@jax.jit
def coo_to_csr(A: jsparse.COO, sum_duplicates: bool = True, result_length: int = 0):
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
        data, rows, cols = coo_arrays_sum_duplicates(A, result_length=result_length)
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
