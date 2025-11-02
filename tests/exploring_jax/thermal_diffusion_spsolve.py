import jax

#jax.config.update("jax_default_device", jax.devices("cpu")[0])
#jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.extend
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

import inspect
import math
from functools import partial

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

verbose = True

###################################################################################################
# Helper functions


def debug_print(x):
    prev_frame = inspect.currentframe().f_back
    prev_fame_info = inspect.getframeinfo(prev_frame)
    callers_local_vars = prev_frame.f_locals.items()
    x_names = [var_name for var_name, var_val in callers_local_vars if var_val is x]
    x_name = x_names[0] if len(x_names) > 0 else "<non-named value>"
    jax.debug.print(
        "From {d} line {e}:\n {a}, shape={b} = \n{c}",
        a=x_name,
        b=x.shape,
        c=x,
        d=prev_fame_info.filename,
        e=prev_fame_info.lineno,
    )


def empty_print(x):
    pass


if verbose:
    dprint = debug_print
else:
    dprint = empty_print


def phi_at_xi(xi):
    """
    Evaluates the two linear basis (hat) functions for a single element at
    a given parametric coordinate, xi, which should span [-1, 1].
    """
    return jnp.array(
        [
            (1.0 - xi) / 2.0,  # basis function 0
            (xi + 1.0) / 2.0,  # basis function 1
        ]
    )


def phi_at_x(x, x_n):
    """
    Evaluates the two linear basis (hat) functions for a single element at a given coordinate,
    x, which should be between the nodes of the element x_n[0] and x_n[1].
    """
    return jnp.array(
        [
            (x_n[1] - x) / (x_n[1] - x_n[0]),  # basis function 0
            (x - x_n[0]) / (x_n[1] - x_n[0]),  # basis function 1
        ]
    )


# Computes the derivative of the basis functions with respect to x.
# Note: vmap's over a list of values of 'x' but the second argument 'x_n" is not mapped over.
dphi_dx_at_x = jax.vmap(jax.jacfwd(phi_at_x), in_axes=(0, None))


@jax.jit
def element_residual(k_func, x_n, T_n, xi_q, W_q):
    """
    Evaluates the main term in the thermal diffusion governing equation for a single element:
    integral of d(k(T) * dT/dx)/dx from x_n[0] to x_n[-1]
    """

    # Each basis function, n, (column) evaluated at each quadrature point, q, (row)
    phi_qn = phi_at_xi(xi_q).T

    # Coordinate of each quadrature point
    x_q = jnp.einsum("qn,n->q", phi_qn, x_n)
    # debug_print(x_q)

    # Temperature at each quadrature point
    T_q = jnp.einsum("qn,n->q", phi_qn, T_n)
    # debug_print(T_q)

    # Derivative of each basis function, n, (column) with respect to x evaluated at each quadrature
    # point, q (row).
    dphi_dx_qn = dphi_dx_at_x(x_q, x_n)
    # debug_print(dphi_dx_qn)

    # Derivative of T w.r.t. x at each quadrature point, q.
    dT_dx_q = jnp.einsum("qn,n->q", dphi_dx_qn, T_n)
    # debug_print(dT_dx_q)

    # Coefficient of thermal expansion evaluated at each quadrature point for the respective T
    k_q = k_func(T_q)

    # Primary term for thermal diffusion governing equation, integral of d(k(T) * dT/dx)/dx
    # over the element.
    dx_k_dT_dx = jnp.einsum("qn,q->n", dphi_dx_qn, k_q * dT_dx_q * W_q)
    # debug_print(dx_k_dT_dx)

    return dx_k_dT_dx


# A vmap'ed version of the element residual function that maps over the elements
R_vmap = jax.vmap(element_residual, in_axes=(None, 0, 0, None, None))


@partial(jax.jit, static_argnums=(0,))
def get_assembly_matrix(n_points: int) -> jsparse.COO:
    """
    Create a sparse matrix that acts as an assembly map in that the matrix multiplication with
    an unraveled elementwise vector produces the assembled vector.

    Note: assumes points are ordered from left to right (ascending values of x).
    Note: only valid for 1D.
    """

    I = jnp.array(range(n_points))
    I = jnp.vstack([I[0:-1], I[1:]]).T
    rows = I.ravel()
    # debug_print(rows)
    cols = jnp.array(range(rows.shape[0]))
    # debug_print(cols)
    data = jnp.ones_like(cols)
    assembly_matrix = jsparse.COO(
        (data, rows, cols), shape=(n_points, 2 * (n_points - 1))
    )
    return assembly_matrix


@jax.jit
def assembled_residual(k_func, points, T, xi_q, W_q, assembly_matrix):
    """
    Evaluates the element_residual function for each element, assembles the element contributions
    into a "global" residual vector, and returns it.

    Note: it does not inlcude the Dirichlet boundary conditions yet, apply_dirichlet_bcs.
    """
    # Coordinates of points arranged by element (row) then point / node (column)
    x_en = jnp.vstack([points[0:-1], points[1:]]).T

    # Temperature at points arranged by element (row) then point / node (column)
    T_en = jnp.vstack([T[0:-1], T[1:]]).T

    # Evaluates the residual for all elements
    R_en = R_vmap(k_func, x_en, T_en, xi_q, W_q)

    # Assemble the residual vector
    R_assembled = assembly_matrix @ R_en.ravel()
    return R_assembled


@jax.jit
def assembled_jacobian(k_func, points, T, xi_q, W_q) -> jsparse.COO:
    """
    Evaluates the Jacobian, i.e. the derivative of the residual function with respect to
    the solution (temperature), for each element, assembles the element contributions
    into a "global" sparse matrix, and returns it.

    Note: it does not inlcude the Dirichlet boundary conditions yet, apply_dirichlet_bcs.
    """
    # A function that evaluates the Jacobian, i.e. the derivative of the residual function
    # with respect to the solution (temperature).
    # Note: A vmap'ed version of the Jacobian that maps over the elements.
    J_vmap = jax.vmap(
        jax.jacfwd(element_residual, argnums=2), in_axes=(None, 0, 0, None, None)
    )

    # Coordinates of points arranged by element (row) then point / node (column)
    x_en = jnp.vstack([points[0:-1], points[1:]]).T

    # Temperature at points arranged by element (row) then point / node (column)
    T_en = jnp.vstack([T[0:-1], T[1:]]).T

    # Evaluates the Jacobian for all elements
    J_enn = J_vmap(k_func, x_en, T_en, xi_q, W_q)
    debug_print(J_enn)

    # Assemble the Jacobian into a sparse COO matrix
    I = jnp.array(range(points.shape[0]))
    I = jnp.vstack([I[0:-1], I[1:]]).T
    cols, rows = jax.vmap(jnp.meshgrid)(I, I)
    debug_print(rows)
    debug_print(cols)
    # Note: the shape of the matrix assumes there is one DOF per a node.
    J_assembled = jsparse.COO(
        (J_enn.ravel(), rows.ravel(), cols.ravel()),
        shape=(points.shape[0], points.shape[0]),
    )
    J_dense = J_assembled.todense()
    debug_print(J_dense)
    return J_assembled


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
    dprint(keep_mask)

    # Create the new data entries for 1's on the diagon corresponding to the BCs.
    num_bcs = dirichlet_dofs.shape[0]
    bc_data = jnp.ones(num_bcs, dtype=A.dtype)
    dprint(bc_data)

    # Create a new sparse matrix by concatenating the old (filtered) and new (BC) entries
    A_modified = jsparse.COO(
        (
            jnp.concatenate([A.data[keep_mask], bc_data]),
            jnp.concatenate([A.row[keep_mask], dirichlet_dofs]),
            jnp.concatenate([A.col[keep_mask], dirichlet_dofs]),
        ),
        shape=A.shape,
    )
    dprint(A_modified.data)
    A_dense = A_modified.todense()
    dprint(A_dense)

    # Update the RHS vector
    tmp = jnp.zeros_like(b)
    tmp = tmp.at[dirichlet_dofs].set(dirichlet_values)
    b_modified = b - A @ tmp
    b_modified = b_modified.at[dirichlet_dofs].set(dirichlet_values)

    return A_modified, b_modified


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
    A_dense = A_csr.todense()
    debug_print(A_dense)

    return scipy.sparse.linalg.spsolve(A_csr, b)


# @jax.jit
def __solve_gpu(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b for a GPU backend.
    Returns the solution, x.
    """
    # Get the permutation that sorts the matrix entries
    perm = jnp.lexsort((A.col, A.row))
    dprint(perm)

    # Creates an array of (row, col) entries (sorted by row then col using perm)
    sorted_indices = jnp.vstack((A.row[perm], A.col[perm])).T
    # An array of indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(sorted_indices, axis=0)
    dprint(sorted_indices)
    dprint(diff)
    # Boolean mask indicating if each (row, col) value is unique
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))
    dprint(uniq_mask)

    # A map from the unique order to the original order
    unique_indices = perm[uniq_mask]
    # A map from the original order to the unique order
    inv_indices = jnp.zeros_like(perm).at[perm].set(jnp.cumsum(uniq_mask) - 1)
    # Effectively sums duplicates and returns the values in the permuated order
    unique_data = jnp.bincount(inv_indices, weights=A.data)
    unique_cols = A.col[unique_indices]

    # Count the number of non-zero elements in each row.
    # The 'length' argument is crucial to ensure the output array has size num_rows,
    # even if the last rows are empty.
    num_rows, _ = A.shape
    nnz_per_row = jnp.bincount(sorted_indices[unique_indices][:,0], length=num_rows)

    # Build the index pointer array (indptr) from the counts.
    # This is a cumulative sum of the non-zero counts per row.
    # The first element of indptr is always 0.
    indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(nnz_per_row)])

    return jsparse.linalg.spsolve(
        unique_data,
        unique_cols.astype(jnp.int32),
        indptr.astype(jnp.int32),
        b,
        tol=1e-8,
        reorder=1,
    )


def solve_sp(A: jsparse.COO, b: jnp.ndarray):
    """
    Sparse direct solve for system A*x = b.
    Returns the solution, x.
    """
    print(f"Detected Backend: {jax.extend.backend.get_backend().platform}")
    match jax.extend.backend.get_backend().platform:
        case "cpu":
            return __solve_cpu(A, b)
        case "gpu":
            return __solve_gpu(A, b)
    raise Exception(f"Backend {jax.extend.backend.get_backend().platform} unsupported.")


@jax.jit
def residual_norm(
    R: jnp.ndarray,
    T: jnp.ndarray,
    dirichlet_dofs: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
):
    """
    Evaluates the L-2 norm of the residual vector while including Dirichlet BCs.
    """
    return jnp.linalg.norm(
        R.at[dirichlet_dofs].set(T[dirichlet_dofs] - dirichlet_values)
    )


###################################################################################################
# 1D Thermal Diffusion Problem Setup

# Points used to discretize the domain of [0, 1]
N = 3  # Number of points for mesh
points = jnp.linspace(start=0.0, stop=1.0, num=N)
dprint(points)

# 3-point Gauss quadrature points and weights
xi_q = jnp.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)])
dprint(xi_q)
weights_q = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
dprint(weights_q)

# Coefficient of thermal conduction as a function of temperature, W / (m*K)
# Part A: Constant value
# k = lambda T: 50.
# Part B: Nonlinear, power law fit of data for monolithic copper
k = lambda T: 792.15 * T ** (-0.118)
# Testing a constant
# k = jax.tree_util.Partial(lambda T: 380.)
# k = jax.tree_util.Partial(lambda T: 1e6 * (T) ** (-20.) + 100.)

# Transform the function to make it compatible with pytrees and vmap'ed
k = jax.tree_util.Partial(jax.vmap(k))

# Plot the coefficient of thermal conduction
T_plot = jnp.linspace(start=100.0, stop=1000.0, num=100)
plt.plot(np.array(T_plot), np.array(k(T_plot)))
plt.title("Coeff. of Thermal Conduction vs. Temperature")
plt.xlabel("T (K)")
plt.ylabel("k (W / (m*K))")
plt.show()

# Apply a Dirichlet BC to the left hand side (index 0) of T=500K
dirichlet_dofs = jnp.array([0, N - 1])
dirichlet_values = jnp.array([100.0, 1000.0])

# Initial guess for temperature at points
T = jnp.linspace(start=100.0, stop=100.0, num=N)

# Helpful matrix to assemble elementwise vectors into a global vector via a mat-vec multiplication
assembly_matrix = get_assembly_matrix(points.shape[0])

###################################################################################################
# Solver Section

# Evaluates the residual for the initial temperature T_0
R = assembled_residual(
    k_func=k,
    points=points,
    T=T,
    xi_q=xi_q,
    W_q=weights_q,
    assembly_matrix=assembly_matrix,
)
debug_print(R)

T_history = [T]
iteration = 0
R_norm = residual_norm(R, T, dirichlet_dofs, dirichlet_values)
while R_norm > 1e-6 or iteration == 0:

    print(f"Iteration {iteration} |R| = {R_norm}")

    # Evaluates the Jacobian for the initial temperature T_0
    J = assembled_jacobian(k_func=k, points=points, T=T, xi_q=xi_q, W_q=weights_q)
    dprint(J)

    # Make the Dirichlet BCs incremental (i.e. the value of the BC is actually how much needs to be
    # added to the current value of T).
    # Note: System of equations for Newton type solve is: J(T) * delta_T = -R(T)
    lhs_matrix, rhs_vector = apply_dirichlet_bcs(
        J, -R, dirichlet_dofs, dirichlet_values - T[dirichlet_dofs]
    )

    # Solve for the increment of temperature
    delta_T = solve_sp(lhs_matrix, rhs_vector)
    dprint(delta_T)

    # Update the temperature given the increment.
    T += delta_T
    debug_print(T)

    T_history.append(T)

    R = assembled_residual(
        k_func=k,
        points=points,
        T=T,
        xi_q=xi_q,
        W_q=weights_q,
        assembly_matrix=assembly_matrix,
    )
    R_norm = residual_norm(R, T, dirichlet_dofs, dirichlet_values)

    iteration += 1

    break

print(f"Final Iteration {iteration} |R| = {R_norm}")

viridis_cmap = plt.get_cmap("viridis_r", len(T_history))
for i, T_i in enumerate(T_history):
    plt.plot(points, T_i, color=viridis_cmap(i), label=f"NL Iteration {i}")
plt.legend()
plt.title("Temperature Profile for Nonlinear Iterations")
plt.xlabel("x (m)")
plt.ylabel("T (K)")
plt.show()
