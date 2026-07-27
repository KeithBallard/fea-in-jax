import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
import warp as wp

wp.init()

@wp.kernel
def apply_mpc_to_jacobian_warp_kernel_f32(
    K_row: wp.array(dtype=int),
    K_col: wp.array(dtype=int),
    K_data: wp.array(dtype=wp.float32),
    dep_to_constraint_idx: wp.array(dtype=int),
    dep_cols: wp.array(dtype=int, ndim=2),
    dep_weights: wp.array(dtype=wp.float32, ndim=2),
    out_row: wp.array(dtype=int),
    out_col: wp.array(dtype=int),
    out_data: wp.array(dtype=wp.float32),
):
    idx = wp.tid()
    if idx >= K_row.shape[0]:
        return
        
    r = K_row[idx]
    c = K_col[idx]
    val = K_data[idx]
    
    r_local = dep_to_constraint_idx[r]
    c_local = dep_to_constraint_idx[c]
    
    r_is_fpc = False
    if r_local != -1:
        if dep_cols[r_local, 0] == -1:
            r_is_fpc = True
            
    c_is_fpc = False
    if c_local != -1:
        if dep_cols[c_local, 0] == -1:
            c_is_fpc = True
            
    out_offset = idx * 4
    
    if r_is_fpc or c_is_fpc:
        # Fixed point constraints are zeroed out and do not couple to other terms
        for i in range(4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = 0.0
    elif r_local == -1 and c_local == -1:
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
        
        out_row[out_offset] = r0
        out_col[out_offset] = c0
        out_data[out_offset] = val * rw0 * cw0
        
        if c1 != -1:
            out_row[out_offset + 1] = r0
            out_col[out_offset + 1] = c1
            out_data[out_offset + 1] = val * rw0 * cw1
        else:
            out_row[out_offset + 1] = -1
            out_col[out_offset + 1] = -1
            out_data[out_offset + 1] = 0.0
            
        if r1 != -1:
            out_row[out_offset + 2] = r1
            out_col[out_offset + 2] = c0
            out_data[out_offset + 2] = val * rw1 * cw0
        else:
            out_row[out_offset + 2] = -1
            out_col[out_offset + 2] = -1
            out_data[out_offset + 2] = 0.0
            
        if r1 != -1 and c1 != -1:
            out_row[out_offset + 3] = r1
            out_col[out_offset + 3] = c1
            out_data[out_offset + 3] = val * rw1 * cw1
        else:
            out_row[out_offset + 3] = -1
            out_col[out_offset + 3] = -1
            out_data[out_offset + 3] = 0.0


@wp.kernel
def apply_mpc_to_jacobian_warp_kernel_f64(
    K_row: wp.array(dtype=int),
    K_col: wp.array(dtype=int),
    K_data: wp.array(dtype=wp.float64),
    dep_to_constraint_idx: wp.array(dtype=int),
    dep_cols: wp.array(dtype=int, ndim=2),
    dep_weights: wp.array(dtype=wp.float64, ndim=2),
    out_row: wp.array(dtype=int),
    out_col: wp.array(dtype=int),
    out_data: wp.array(dtype=wp.float64),
):
    idx = wp.tid()
    if idx >= K_row.shape[0]:
        return
        
    r = K_row[idx]
    c = K_col[idx]
    val = K_data[idx]
    
    r_local = dep_to_constraint_idx[r]
    c_local = dep_to_constraint_idx[c]
    
    r_is_fpc = False
    if r_local != -1:
        if dep_cols[r_local, 0] == -1:
            r_is_fpc = True
            
    c_is_fpc = False
    if c_local != -1:
        if dep_cols[c_local, 0] == -1:
            c_is_fpc = True
            
    out_offset = idx * 4
    
    if r_is_fpc or c_is_fpc:
        # Fixed point constraints are zeroed out and do not couple to other terms
        for i in range(4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = wp.float64(0.0)
    elif r_local == -1 and c_local == -1:
        # Term 1: Indep - Indep
        out_row[out_offset] = r
        out_col[out_offset] = c
        out_data[out_offset] = val
        for i in range(1, 4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = wp.float64(0.0)
            
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
            out_data[out_offset + 1] = wp.float64(0.0)
            
        for i in range(2, 4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = wp.float64(0.0)
            
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
            out_data[out_offset + 1] = wp.float64(0.0)
            
        for i in range(2, 4):
            out_row[out_offset + i] = -1
            out_col[out_offset + i] = -1
            out_data[out_offset + i] = wp.float64(0.0)
            
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
        
        out_row[out_offset] = r0
        out_col[out_offset] = c0
        out_data[out_offset] = val * rw0 * cw0
        
        if c1 != -1:
            out_row[out_offset + 1] = r0
            out_col[out_offset + 1] = c1
            out_data[out_offset + 1] = val * rw0 * cw1
        else:
            out_row[out_offset + 1] = -1
            out_col[out_offset + 1] = -1
            out_data[out_offset + 1] = wp.float64(0.0)
            
        if r1 != -1:
            out_row[out_offset + 2] = r1
            out_col[out_offset + 2] = c0
            out_data[out_offset + 2] = val * rw1 * cw0
        else:
            out_row[out_offset + 2] = -1
            out_col[out_offset + 2] = -1
            out_data[out_offset + 2] = wp.float64(0.0)
            
        if r1 != -1 and c1 != -1:
            out_row[out_offset + 3] = r1
            out_col[out_offset + 3] = c1
            out_data[out_offset + 3] = val * rw1 * cw1
        else:
            out_row[out_offset + 3] = -1
            out_col[out_offset + 3] = -1
            out_data[out_offset + 3] = wp.float64(0.0)


@wp.kernel
def populate_diag_kernel_f32(
    dep_dofs: wp.array(dtype=int),
    out_row: wp.array(dtype=int),
    out_col: wp.array(dtype=int),
    out_data: wp.array(dtype=wp.float32),
    offset: int,
):
    idx = wp.tid()
    if idx >= dep_dofs.shape[0]:
        return
    dof = dep_dofs[idx]
    out_row[offset + idx] = dof
    out_col[offset + idx] = dof
    out_data[offset + idx] = 1.0


@wp.kernel
def populate_diag_kernel_f64(
    dep_dofs: wp.array(dtype=int),
    out_row: wp.array(dtype=int),
    out_col: wp.array(dtype=int),
    out_data: wp.array(dtype=wp.float64),
    offset: int,
):
    idx = wp.tid()
    if idx >= dep_dofs.shape[0]:
        return
    dof = dep_dofs[idx]
    out_row[offset + idx] = dof
    out_col[offset + idx] = dof
    out_data[offset + idx] = wp.float64(1.0)


def apply_mpc_to_jacobian_warp(K: jsparse.COO, constraints) -> jsparse.COO:
    """
    Applies multi-point constraints (MPCs) and fixed-point (Dirichlet) constraints
    to a sparse Jacobian matrix in COO format using extremely fast, GPU-native NVidia Warp kernels.
    """
    num_dofs = K.shape[0]
    nnz_K = K.data.shape[0]
    n_dep = constraints.dep_dofs.shape[0]

    dep_dofs_np = np.array(constraints.dep_dofs)
    dep_to_constraint_idx_np = np.full((num_dofs,), -1, dtype=np.int32)
    for i, dof in enumerate(dep_dofs_np):
        dep_to_constraint_idx_np[dof] = i

    P_indices_np = np.array(constraints.P.indices)
    P_data_np = np.array(constraints.P.data)

    dep_cols_np = np.full((n_dep, 2), -1, dtype=np.int32)
    dep_weights_np = np.zeros((n_dep, 2), dtype=np.float64)

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

    out_size = nnz_K * 4 + n_dep
    device = "cuda" if wp.is_cuda_available() else "cpu"

    wp_dtype = wp.float64 if K.data.dtype == jnp.float64 else wp.float32

    K_row_wp = wp.array(np.array(K.row), dtype=int, device=device)
    K_col_wp = wp.array(np.array(K.col), dtype=int, device=device)
    K_data_wp = wp.array(np.array(K.data), dtype=wp_dtype, device=device)

    dep_to_constraint_idx_wp = wp.array(dep_to_constraint_idx_np, dtype=int, device=device)
    dep_cols_wp = wp.array(dep_cols_np, dtype=int, device=device)
    dep_weights_wp = wp.array(dep_weights_np, dtype=wp_dtype, device=device)
    dep_dofs_wp = wp.array(dep_dofs_np, dtype=int, device=device)

    out_row_wp = wp.empty(out_size, dtype=int, device=device)
    out_col_wp = wp.empty(out_size, dtype=int, device=device)
    out_data_wp = wp.empty(out_size, dtype=wp_dtype, device=device)

    if K.data.dtype == jnp.float64:
        kernel_proj = apply_mpc_to_jacobian_warp_kernel_f64
        kernel_diag = populate_diag_kernel_f64
    else:
        kernel_proj = apply_mpc_to_jacobian_warp_kernel_f32
        kernel_diag = populate_diag_kernel_f32

    wp.launch(
        kernel=kernel_proj,
        dim=nnz_K,
        inputs=[
            K_row_wp, K_col_wp, K_data_wp,
            dep_to_constraint_idx_wp,
            dep_cols_wp, dep_weights_wp,
            out_row_wp, out_col_wp, out_data_wp
        ],
        device=device
    )

    if n_dep > 0:
        wp.launch(
            kernel=kernel_diag,
            dim=n_dep,
            inputs=[
                dep_dofs_wp,
                out_row_wp, out_col_wp, out_data_wp,
                nnz_K * 4
            ],
            device=device
        )

    out_row_np = out_row_wp.numpy()
    out_col_np = out_col_wp.numpy()
    out_data_np = out_data_wp.numpy()

    valid_mask = (out_row_np != -1)
    filtered_row = out_row_np[valid_mask]
    filtered_col = out_col_np[valid_mask]
    filtered_data = out_data_np[valid_mask]

    return jsparse.COO(
        (jnp.array(filtered_data, dtype=K.data.dtype), jnp.array(filtered_row), jnp.array(filtered_col)),
        shape=K.shape,
        rows_sorted=False,
        cols_sorted=False
    )
