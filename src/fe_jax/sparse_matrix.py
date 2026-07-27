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


def _apply_mpc_to_jacobian_jax(K: jsparse.COO, constraints) -> jsparse.COO:
    """
    Applies multi-point constraints (MPCs) and fixed-point (Dirichlet) constraints
    to a sparse Jacobian matrix in COO format using fully vectorized JAX/GPU operations.
    """
    dep_dofs = constraints.dep_dofs
    n_total_dofs = K.shape[0]

    P_indices = constraints.P.indices  # shape (nse, 2)
    P_data = constraints.P.data        # shape (nse,)

    row_P = dep_dofs[P_indices[:, 0]]
    col_P = P_indices[:, 1]
    data_P = P_data

    K_row = K.row
    K_col = K.col
    K_data = K.data

    is_row_dep = jnp.isin(K_row, dep_dofs)
    is_col_dep = jnp.isin(K_col, dep_dofs)
    mask_1 = ~(is_row_dep | is_col_dep)
    row_1 = K_row
    col_1 = K_col
    data_1 = jnp.where(mask_1, K_data, 0.0)

    def coo_matmul(A_row, A_col, A_data, B_row, B_col, B_data):
        col_A_expanded = A_col[:, None]
        row_B_expanded = B_row[None, :]
        match_mask = (col_A_expanded == row_B_expanded)
        idx_A, idx_B = jnp.where(match_mask)

        row_res = A_row[idx_A]
        col_res = B_col[idx_B]
        data_res = A_data[idx_A] * B_data[idx_B]
        return row_res, col_res, data_res

    def coo_matmul_transpose_left(B_row, B_col, B_data, C_row, C_col, C_data):
        row_B_expanded = B_row[:, None]
        row_C_expanded = C_row[None, :]
        match_mask = (row_B_expanded == row_C_expanded)
        idx_B, idx_C = jnp.where(match_mask)

        row_res = B_col[idx_B]
        col_res = C_col[idx_C]
        data_res = B_data[idx_B] * C_data[idx_C]
        return row_res, col_res, data_res

    if P_indices.shape[0] > 0:
        row_2_raw, col_2_raw, data_2_raw = coo_matmul(K_row, K_col, K_data, row_P, col_P, data_P)
        is_row_dep_2 = jnp.isin(row_2_raw, dep_dofs)
        row_2 = row_2_raw[~is_row_dep_2]
        col_2 = col_2_raw[~is_row_dep_2]
        data_2 = data_2_raw[~is_row_dep_2]

        mask_K_indep = ~jnp.isin(K_col, dep_dofs)
        row_K_indep = K_row
        col_K_indep = K_col
        data_K_indep = jnp.where(mask_K_indep, K_data, 0.0)
        row_3, col_3, data_3 = coo_matmul_transpose_left(row_P, col_P, data_P, row_K_indep, col_K_indep, data_K_indep)

        row_4, col_4, data_4 = coo_matmul_transpose_left(row_P, col_P, data_P, row_2_raw, col_2_raw, data_2_raw)
    else:
        row_2 = row_3 = row_4 = jnp.zeros((0,), dtype=jnp.int32)
        col_2 = col_3 = col_4 = jnp.zeros((0,), dtype=jnp.int32)
        data_2 = data_3 = data_4 = jnp.zeros((0,), dtype=jnp.float32)

    row_diag = dep_dofs
    col_diag = dep_dofs
    data_diag = jnp.ones_like(dep_dofs, dtype=jnp.float32)

    total_rows = jnp.concatenate([row_1, row_2, row_3, row_4, row_diag])
    total_cols = jnp.concatenate([col_1, col_2, col_3, col_4, col_diag])
    total_data = jnp.concatenate([data_1, data_2, data_3, data_4, data_diag])

    return jsparse.COO(
        (total_data, total_rows, total_cols),
        shape=K.shape,
        rows_sorted=False,
        cols_sorted=False
    )


def apply_mpc_to_jacobian(K: jsparse.COO, constraints) -> jsparse.COO:
    """
    Applies multi-point constraints (MPCs) and fixed-point (Dirichlet) constraints
    to a sparse Jacobian matrix in COO format.
    """
    dep_dofs = constraints.dep_dofs
    if dep_dofs.shape[0] == 0:
        return K

    return _apply_mpc_to_jacobian_jax(K, constraints)

