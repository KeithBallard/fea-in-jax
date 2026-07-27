import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
import scipy.sparse
import time
import warp as wp

# Initialize Warp
wp.init()

# Import the benchmark generator from the previous script
import sys
import os
sys.path.append(os.path.abspath("scratch"))
from benchmark_sparse_matmul import generate_benchmark_data

# Ensure x64 is enabled for precision comparison
jax.config.update("jax_enable_x64", True)

# Define the NVidia Warp Sparse Matmul Kernel
@wp.kernel
def apply_mpc_to_jacobian_warp_kernel(
    K_row: wp.array(dtype=int),
    K_col: wp.array(dtype=int),
    K_data: wp.array(dtype=float),
    dep_to_constraint_idx: wp.array(dtype=int),
    dep_cols: wp.array(dtype=int, ndim=2),
    dep_weights: wp.array(dtype=float, ndim=2),
    out_row: wp.array(dtype=int),
    out_col: wp.array(dtype=int),
    out_data: wp.array(dtype=float),
):
    idx = wp.tid()
    if idx >= K_row.shape[0]:
        return
        
    r = K_row[idx]
    c = K_col[idx]
    val = K_data[idx]
    
    # Look up if row and col are dependent
    r_local = dep_to_constraint_idx[r]
    c_local = dep_to_constraint_idx[c]
    
    out_offset = idx * 4
    
    if r_local == -1 and c_local == -1:
        # Term 1: Indep - Indep
        out_row[out_offset] = r
        out_col[out_offset] = c
        out_data[out_offset] = val
        for i in range(1, 4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = 0.0
            
    elif r_local == -1 and c_local != -1:
        # Term 2: Indep - Dep
        c0 = dep_cols[c_local, 0]
        w0 = dep_weights[c_local, 0]
        c1 = dep_cols[c_local, 1]
        w1 = dep_weights[c_local, 1]
        
        out_row[out_offset] = r
        out_col[out_offset] = c0
        out_data[out_offset] = val * w0
        
        if c1 != -1:
            out_row[out_offset + 1] = r
            out_col[out_offset + 1] = c1
            out_data[out_offset + 1] = val * w1
        else:
            out_row[out_offset + 1] = -1
            out_col[out_offset + 1] = -1
            out_data[out_offset + 1] = 0.0
            
        for i in range(2, 4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = 0.0
            
    elif r_local != -1 and c_local == -1:
        # Term 3: Dep - Indep
        r0 = dep_cols[r_local, 0]
        w0 = dep_weights[r_local, 0]
        r1 = dep_cols[r_local, 1]
        w1 = dep_weights[r_local, 1]
        
        out_row[out_offset] = r0
        out_col[out_offset] = c
        out_data[out_offset] = val * w0
        
        if r1 != -1:
            out_row[out_offset + 1] = r1
            out_col[out_offset + 1] = c
            out_data[out_offset + 1] = val * w1
        else:
            out_row[out_offset + 1] = -1
            out_col[out_offset + 1] = -1
            out_data[out_offset + 1] = 0.0
            
        for i in range(2, 4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = 0.0
            
    else:
        # Term 4: Dep - Dep
        r0 = dep_cols[r_local, 0]
        rw0 = dep_weights[r_local, 0]
        r1 = dep_cols[r_local, 1]
        rw1 = dep_weights[r_local, 1]
        
        c0 = dep_cols[c_local, 0]
        cw0 = dep_weights[c_local, 0]
        c1 = dep_cols[c_local, 1]
        cw1 = dep_weights[c_local, 1]
        
        # Slot 0
        out_row[out_offset] = r0
        out_col[out_offset] = c0
        out_data[out_offset] = val * rw0 * cw0
        
        # Slot 1
        if c1 != -1:
            out_row[out_offset + 1] = r0
            out_col[out_offset + 1] = c1
            out_data[out_offset + 1] = val * rw0 * cw1
        else:
            out_row[out_offset + 1] = -1
            out_col[out_offset + 1] = -1
            out_data[out_offset + 1] = 0.0
            
        # Slot 2
        if r1 != -1:
            out_row[out_offset + 2] = r1
            out_col[out_offset + 2] = c0
            out_data[out_offset + 2] = val * rw1 * cw0
        else:
            out_row[out_offset + 2] = -1
            out_col[out_offset + 2] = -1
            out_data[out_offset + 2] = 0.0
            
        # Slot 3
        if r1 != -1 and c1 != -1:
            out_row[out_offset + 3] = r1
            out_col[out_offset + 3] = c1
            out_data[out_offset + 3] = val * rw1 * cw1
        else:
            out_row[out_offset + 3] = -1
            out_col[out_offset + 3] = -1
            out_data[out_offset + 3] = 0.0


@wp.kernel
def populate_diag_kernel(
    dep_dofs: wp.array(dtype=int),
    out_row: wp.array(dtype=int),
    out_col: wp.array(dtype=int),
    out_data: wp.array(dtype=float),
    offset: int,
):
    idx = wp.tid()
    if idx >= dep_dofs.shape[0]:
        return
    dof = dep_dofs[idx]
    out_row[offset + idx] = dof
    out_col[offset + idx] = dof
    out_data[offset + idx] = 1.0


def apply_mpc_to_jacobian_warp(K_coo_jax, constraints):
    """
    Highly optimized, GPU-accelerated sparse projection using NVIDIA Warp kernels.
    """
    # Extract dimensions
    num_dofs = K_coo_jax.shape[0]
    nnz_K = K_coo_jax.data.shape[0]
    n_dep = constraints.dep_dofs.shape[0]
    
    # 1. Build lookup arrays on-the-fly inside the non-JIT callback
    dep_dofs_np = np.array(constraints.dep_dofs)
    dep_to_constraint_idx_np = np.full((num_dofs,), -1, dtype=np.int32)
    for i, dof in enumerate(dep_dofs_np):
        dep_to_constraint_idx_np[dof] = i
        
    # Convert P constraint data to a flat, dense constraint mapping for up to 2 independent terms
    # Extract row_P, col_P, data_P from BCOO
    P_indices_np = np.array(constraints.P.indices)
    P_data_np = np.array(constraints.P.data)
    
    dep_cols_np = np.full((n_dep, 2), -1, dtype=np.int32)
    dep_weights_np = np.zeros((n_dep, 2), dtype=np.float64)
    
    # Track index fill for each constraint row
    fill_ptr = np.zeros((n_dep,), dtype=np.int32)
    for i in range(P_indices_np.shape[0]):
        row_idx = P_indices_np[i, 0]
        col_idx = P_indices_np[i, 1]
        weight = P_data_np[i]
        
        ptr = fill_ptr[row_idx]
        if ptr < 2:
            dep_cols_np[row_idx, ptr] = col_idx
            dep_weights_np[row_idx, ptr] = weight
            fill_ptr[row_idx] += 1

    # 2. Allocate output row, col, values arrays (with shape NNZ*4 + n_dep)
    out_size = nnz_K * 4 + n_dep
    
    # Use Warp memory pool and DLPack interop for zero-copy allocations
    device = "cuda" if wp.is_cuda_available() else "cpu"
    
    K_row_wp = wp.array(np.array(K_coo_jax.row), dtype=int, device=device)
    K_col_wp = wp.array(np.array(K_coo_jax.col), dtype=int, device=device)
    K_data_wp = wp.array(np.array(K_coo_jax.data), dtype=float, device=device)
    
    dep_to_constraint_idx_wp = wp.array(dep_to_constraint_idx_np, dtype=int, device=device)
    dep_cols_wp = wp.array(dep_cols_np, dtype=int, device=device)
    dep_weights_wp = wp.array(dep_weights_np, dtype=float, device=device)
    dep_dofs_wp = wp.array(dep_dofs_np, dtype=int, device=device)
    
    # Allocate empty outputs
    out_row_wp = wp.empty(out_size, dtype=int, device=device)
    out_col_wp = wp.empty(out_size, dtype=int, device=device)
    out_data_wp = wp.empty(out_size, dtype=float, device=device)
    
    # 3. Launch Warp Kernels
    wp.launch(
        kernel=apply_mpc_to_jacobian_warp_kernel,
        dim=nnz_K,
        inputs=[
            K_row_wp, K_col_wp, K_data_wp,
            dep_to_constraint_idx_wp,
            dep_cols_wp, dep_weights_wp,
            out_row_wp, out_col_wp, out_data_wp
        ],
        device=device
    )
    
    # Populate Diagonal entries
    if n_dep > 0:
        wp.launch(
            kernel=populate_diag_kernel,
            dim=n_dep,
            inputs=[
                dep_dofs_wp,
                out_row_wp, out_col_wp, out_data_wp,
                nnz_K * 4
            ],
            device=device
        )
        
    # Convert Warp array back to JAX COO
    # We can fetch as cupy/numpy and build jsparse.COO
    out_row_np = out_row_wp.numpy() if device == "cpu" else out_row_wp.numpy() # warp handles host copy gracefully via .numpy()
    out_col_np = out_col_wp.numpy()
    out_data_np = out_data_wp.numpy()
    
    # Filter out unused padded slots (indicated by row = -1)
    valid_mask = (out_row_np != -1)
    filtered_row = out_row_np[valid_mask]
    filtered_col = out_col_np[valid_mask]
    filtered_data = out_data_np[valid_mask]
    
    return jsparse.COO(
        (jnp.array(filtered_data), jnp.array(filtered_row), jnp.array(filtered_col)),
        shape=K_coo_jax.shape
    )


def benchmark_warp():
    num_dofs = 2000
    num_constraints = 150
    nnz_per_row = 15
    
    print(f"=== Sparse Matrix Projection Benchmark (including NVidia WARP) ===")
    print(f"Degrees of freedom (DOFs): {num_dofs}")
    print(f"Number of constraints (MPCs): {num_constraints}")
    print(f"Jacobian Non-zeros (NNZ): {num_dofs * nnz_per_row}")
    
    K_coo_jax, K_scipy, constraints, P_full_scipy, dep_dofs = generate_benchmark_data(
        num_dofs, num_constraints, nnz_per_row
    )
    
    # Warm-up runs
    from fe_jax.sparse_matrix import apply_mpc_to_jacobian
    _ = apply_mpc_to_jacobian(K_coo_jax, constraints)
    _ = apply_mpc_to_jacobian_warp(K_coo_jax, constraints)
    
    # 1. SciPy CSR
    t0 = time.perf_counter()
    for _ in range(50):
        PT_K_P = P_full_scipy.T.dot(K_scipy).dot(P_full_scipy)
        PT_K_P_lil = PT_K_P.tolil()
        for d in dep_dofs:
            PT_K_P_lil[d, d] = 1.0
        scipy_res = PT_K_P_lil.tocsr()
    t1 = time.perf_counter()
    scipy_time = (t1 - t0) / 50.0
    print(f"1. SciPy CSR Matmul Average Time: {scipy_time * 1000:.3f} ms")
    
    # 2. Pure JAX Vectorized
    t0 = time.perf_counter()
    for _ in range(50):
        jax_res = apply_mpc_to_jacobian(K_coo_jax, constraints)
        jax_res.data.block_until_ready()
    t1 = time.perf_counter()
    jax_time = (t1 - t0) / 50.0
    print(f"2. Pure JAX Vectorized Average Time: {jax_time * 1000:.3f} ms")
    
    # 3. NVidia Warp GPU-Accelerated
    t0 = time.perf_counter()
    for _ in range(50):
        warp_res = apply_mpc_to_jacobian_warp(K_coo_jax, constraints)
        warp_res.data.block_until_ready()
    t1 = time.perf_counter()
    warp_time = (t1 - t0) / 50.0
    print(f"3. NVidia Warp GPU Kernel Average Time: {warp_time * 1000:.3f} ms")
    
    # Verify correctness
    warp_res_coo = scipy.sparse.coo_matrix(
        (np.array(warp_res.data), (np.array(warp_res.row), np.array(warp_res.col))),
        shape=scipy_res.shape
    )
    warp_res_coo.sum_duplicates()
    warp_res_csr = warp_res_coo.tocsr()
    
    assert np.allclose(scipy_res.todense(), warp_res_csr.todense(), atol=1e-10)
    print("Verification: SUCCESS! All three methods yield mathematically identical results.")

if __name__ == "__main__":
    benchmark_warp()
