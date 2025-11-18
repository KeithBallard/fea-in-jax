import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from functools import partial

from .utils import debug_print


def coo_arrays_sum_duplicates(A: jsparse.COO) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns the row-then-column sorted arrays for a new COO matrix after summing
    duplicate indices.

    NOTE NOT JIT-compatible since the length of the resultant array are unknown at compilation.

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
    A: jsparse.COO, result_length: int
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
    # debug_print(sorted_indices)
    # An array of sorted_indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(sorted_indices, axis=0)
    # debug_print(diff)
    # Boolean mask indicating if each (row, col) value is unique, shape=A.col.shape
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))
    # debug_print(uniq_mask)
    # A map from the unique order to the original order
    # NOTE: there is a trick here to get the unique indices while also guaranteeing array sizes
    unique_indices = jnp.sort(jnp.where(uniq_mask, perm, jnp.max(perm) + 1))[
        0:result_length
    ]
    # debug_print(unique_indices)
    # A map from the original order to the unique order
    inv_indices = jnp.zeros_like(perm).at[perm].set(jnp.cumsum(uniq_mask) - 1)
    # debug_print(inv_indices)
    # Effectively sums duplicates and returns the values in the permuated order
    data = jnp.bincount(inv_indices, weights=A.data, length=result_length)
    rows = A.row[unique_indices]
    cols = A.col[unique_indices]
    # debug_print(data)
    # debug_print(rows)
    # debug_print(cols)
    return (data, rows, cols)


@partial(jax.jit, static_argnames=["result_length"])
def coo_sum_duplicates(A: jsparse.COO, result_length: int) -> jsparse.COO:
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
def coo_to_csr(A: jsparse.COO):
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
    jax.debug.print("indptr: {}", indptr)

    return jsparse.CSR((data, cols, indptr), shape=A.shape)



def apply_dirichlet_bcs_lhs(A: jsparse.COO, dirichlet_dofs: jnp.ndarray) -> jsparse.COO:
    """
    Returns a modified COO sparse matrix that has the same sparsity structure as A but modifies
    entries for in-place elimination of Dirichlet BCs, i.e. zero rows/columns and one on the
    diagonal for constrained DoFs.
    """

    # Create a mask that indicates if an index is on a constrained row / column
    row_constrained_mask = jnp.isin(A.row, dirichlet_dofs)
    col_constrained_mask = jnp.isin(A.col, dirichlet_dofs)
    # debug_print(row_constrained_mask)
    # debug_print(col_constrained_mask)
    # Set all values on constrained rows / columns to 0, then set those diagonal terms to 1.
    modified_data = jnp.where(
        ~(row_constrained_mask | col_constrained_mask), A.data, 0.0
    )
    # debug_print(modified_data)
    modified_data = jnp.where(
        (A.row == A.col) & row_constrained_mask, 1.0, modified_data
    )
    # debug_print(modified_data)

    return jsparse.COO(
        (modified_data, A.row, A.col),
        shape=A.shape,
        rows_sorted=A._rows_sorted,
        cols_sorted=A._cols_sorted,
    )


def apply_dirichlet_bcs_rhs(
    A: jsparse.COO,
    b: jnp.ndarray,
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
) -> jnp.ndarray:
    """
    Returns a modified RHS vector for in-place elimination of Dirichlet BCs.

    NOTE residual_w_dirichlet will automatically include this adjustment, so it is not needed in that case!
    """
    tmp = jnp.zeros_like(b)
    tmp = tmp.at[dirichlet_dofs].set(dirichlet_values)
    b_modified = b - A @ tmp
    b_modified = b_modified.at[dirichlet_dofs].set(dirichlet_values)
    return b_modified
