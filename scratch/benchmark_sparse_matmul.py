import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
import scipy.sparse
import time

# Ensure x64 is enabled for precision comparison
jax.config.update("jax_enable_x64", True)

# Import our apply_mpc_to_jacobian function
from fe_jax.sparse_matrix import apply_mpc_to_jacobian
from fe_jax.constraint_system import ConstraintSystem

def generate_benchmark_data(num_dofs=2000, num_constraints=100, nnz_per_row=15):
    """
    Generates realistic sparse Jacobian and constraint matrices for FEA benchmarks.
    """
    np.random.seed(42)
    
    # 1. Generate sparse Jacobian K in COO format
    K_rows = []
    K_cols = []
    K_data = []
    for i in range(num_dofs):
        cols = np.random.choice(num_dofs, size=nnz_per_row, replace=False)
        cols = np.unique(np.concatenate([cols, [i]]))  # ensure diagonal is present
        K_rows.extend([i] * len(cols))
        K_cols.extend(cols)
        K_data.extend(np.random.rand(len(cols)) * 10.0)
        
    K_row = np.array(K_rows, dtype=np.int32)
    K_col = np.array(K_cols, dtype=np.int32)
    K_data = np.array(K_data, dtype=np.float64)
    
    K_coo_jax = jsparse.COO((jnp.array(K_data), jnp.array(K_row), jnp.array(K_col)), shape=(num_dofs, num_dofs))
    K_scipy = scipy.sparse.coo_matrix((K_data, (K_row, K_col)), shape=(num_dofs, num_dofs)).tocsr()
    
    # 2. Generate multi-point constraint system
    # We constrain some dependent DOFs to be linear combinations of independent DOFs
    dep_dofs = np.sort(np.random.choice(num_dofs, size=num_constraints, replace=False))
    indep_dofs = np.setdiff1d(np.arange(num_dofs), dep_dofs)
    
    # Constraint coefficients matrix P of shape (num_constraints, num_dofs)
    # where dep_dof_i = sum_j P_{i, j} * indep_dof_j
    P_indices = []
    P_data = []
    for i in range(num_constraints):
        # each dependent DOF is constrained to 2 independent DOFs
        linked_indep = np.random.choice(indep_dofs, size=2, replace=False)
        P_indices.append([i, linked_indep[0]])
        P_indices.append([i, linked_indep[1]])
        P_data.extend([0.5, -0.5])
        
    P_indices = np.array(P_indices, dtype=np.int32)
    P_data = np.array(P_data, dtype=np.float64)
    
    # Pack into a mockup of ConstraintSystem
    # We need constraints.dep_dofs and constraints.P (which is a jsparse.BCOO)
    dep_dofs_jax = jnp.array(dep_dofs, dtype=jnp.int32)
    P_indices_jax = jnp.array(P_indices, dtype=np.int32)
    P_data_jax = jnp.array(P_data, dtype=np.float64)
    
    P_bcoo = jsparse.BCOO((P_data_jax, P_indices_jax), shape=(num_constraints, num_dofs))
    
    class MockConstraintSystem:
        def __init__(self, dep_dofs, P):
            self.dep_dofs = dep_dofs
            self.P = P
            
    constraints = MockConstraintSystem(dep_dofs_jax, P_bcoo)
    
    # 3. Create full projection matrix P_full for SciPy comparison
    # P_full = I_indep + P_dep
    # which is of shape (num_dofs, num_dofs)
    # For independent rows i, P_full[i, i] = 1
    # For dependent rows i, P_full[dep_dof_i, col_j] = P_data
    P_full_scipy = scipy.sparse.eye(num_dofs, dtype=np.float64).tolil()
    for i, dep_dof in enumerate(dep_dofs):
        # zero out the diagonal for dependent DOFs
        P_full_scipy[dep_dof, dep_dof] = 0.0
        # set constraint coefficients
        indices_for_dep = P_indices[P_indices[:, 0] == i]
        data_for_dep = P_data[P_indices[:, 0] == i]
        for idx, val in zip(indices_for_dep[:, 1], data_for_dep):
            P_full_scipy[dep_dof, idx] = val
            
    P_full_scipy = P_full_scipy.tocsr()
    
    return K_coo_jax, K_scipy, constraints, P_full_scipy, dep_dofs

def benchmark():
    num_dofs = 2000
    num_constraints = 150
    nnz_per_row = 15
    
    print(f"=== Sparse-Sparse Matrix Projection Benchmark ===")
    print(f"Degrees of freedom (DOFs): {num_dofs}")
    print(f"Number of constraints (MPCs): {num_constraints}")
    print(f"Jacobian Non-zeros (NNZ): {num_dofs * nnz_per_row}")
    
    K_coo_jax, K_scipy, constraints, P_full_scipy, dep_dofs = generate_benchmark_data(
        num_dofs, num_constraints, nnz_per_row
    )
    
    # Warm-up run for JAX/CPU or JAX eager execution
    _ = apply_mpc_to_jacobian(K_coo_jax, constraints)
    
    # 1. Benchmarking SciPy Sparse Matrix Multiplication
    # K_constrained = P_full^T * K_scipy * P_full + I_dep
    t0 = time.perf_counter()
    for _ in range(50):
        # P_full^T * K * P_full
        PT_K_P = P_full_scipy.T.dot(K_scipy).dot(P_full_scipy)
        # Add diagonal 1s on dependent DOFs
        PT_K_P_lil = PT_K_P.tolil()
        for d in dep_dofs:
            PT_K_P_lil[d, d] = 1.0
        scipy_res = PT_K_P_lil.tocsr()
    t1 = time.perf_counter()
    scipy_time = (t1 - t0) / 50.0
    print(f"SciPy CSR Matmul Average Time: {scipy_time * 1000:.3f} ms")
    
    # 2. Benchmarking Our Vectorized JAX Implementation (eager/non-JIT execution)
    t0 = time.perf_counter()
    for _ in range(50):
        jax_res = apply_mpc_to_jacobian(K_coo_jax, constraints)
        # Force evaluation by checking array representation
        jax_res.data.block_until_ready()
    t1 = time.perf_counter()
    jax_time = (t1 - t0) / 50.0
    print(f"JAX GPU/CPU Vectorized Eager Projection Average Time: {jax_time * 1000:.3f} ms")
    
    # Compare correctness by checking shapes and non-zeros
    print(f"SciPy final matrix shape: {scipy_res.shape}, NNZ after duplicate sum: {scipy_res.nnz}")
    print(f"JAX final matrix shape: {jax_res.shape}, Raw NNZ (before duplicate sum): {jax_res.data.shape[0]}")
    
    # Sum duplicates in SciPy-like way on JAX COO matrix to verify values match
    jax_scipy_coo = scipy.sparse.coo_matrix(
        (np.array(jax_res.data), (np.array(jax_res.row), np.array(jax_res.col))),
        shape=scipy_res.shape
    )
    jax_scipy_coo.sum_duplicates()
    jax_res_csr = jax_scipy_coo.tocsr()
    
    assert np.allclose(scipy_res.todense(), jax_res_csr.todense(), atol=1e-10)
    print("Verification: SUCCESS! JAX vectorized eager projection matches SciPy dense/CSR output perfectly.")

if __name__ == "__main__":
    benchmark()
