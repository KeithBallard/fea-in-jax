import scipy
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sparse_l2_cond_estimate(A,dense=False):
    A = (A+A.T)/2
    if A.shape[0]>1000 & dense == True:
        print('Matrix is larger than 1000x1000, reverting to sparse algorithms even though you requested "dense=False"')
    else:
        if dense:
            return np.linalg.cond(A.todense(),2)
    lam_min = scipy.sparse.linalg.eigsh(A, k=1, which = "SM", return_eigenvectors=False)[0]
    lam_max = scipy.sparse.linalg.eigsh(A, k=1, which = "LM", return_eigenvectors=False)[0]
    return lam_max/lam_min


def sparse_l1_cond_estimate(A,dense=False):
    if A.shape[0]>1000 & dense == True:
        print('Matrix is larger than 1000x1000, reverting to sparse algorithms even though you requested "dense=False"')
    else:
        if dense:
            return np.linalg.cond(A.todense(),1)
    A = A.tocsc()
    lu = scipy.sparse.linalg.splu(A)

    A_norm = abs(A).sum(axis=0).max()

    Ainv = scipy.sparse.linalg.LinearOperator(
        A.shape,
        matvec=lu.solve,
        rmatvec=lambda x: lu.solve(x, trans="T"),
    )

    Ainv_norm_est = scipy.sparse.linalg.onenormest(Ainv)
    return  A_norm * Ainv_norm_est


def read_free_jacobian_coo(f,ts,nl):
    p = f'ts_{ts}/nl_{nl}/GLOBAL_JACOBIAN_COO/'
    print(p)
    n_dofs = f[f'{p}n_dofs'][:][0]
    A = scipy.sparse.coo_matrix(
        (
            f[f'{p}data_wo_constraints'][:],(
                f[f'{p}rows_wo_constraints'][:],
                f[f'{p}cols_wo_constraints'][:]
            )
        ), shape = (n_dofs, n_dofs)
    ).tocsr()
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs,f[f'{p}dep_dofs'][:])
    return A[free_dofs,:][:, free_dofs]

def plot_Jac_cond(db_file,cond_metric=1, dense=False,ax=None, color='k', label=None):
    pseudo_steps = [int(i.strip('ts_')) for i in list(db_file.keys())]
    pseudo_steps.sort()
    nl_steps = []
    for t in pseudo_steps:
        temp = [int(i.strip('nl_')) for i in list(db_file[f'ts_{t}'].keys())]
        temp.sort()
        nl_steps.append(temp[1:])
    C = []
    next_stage=[]
    for pseudo_stage in pseudo_steps:
        if len(nl_steps[pseudo_stage])>1:
            next_stage.append(nl_steps[pseudo_stage][-1])
            for nl_stage in nl_steps[pseudo_stage]:
                A = read_free_jacobian_coo(db_file,pseudo_stage,nl_stage)
                if cond_metric==1:
                    C.append(sparse_l1_cond_estimate(A, dense=dense))
                elif cond_metric==2:
                    C.append(sparse_l2_cond_estimate(A, dense=dense))
                else:
                    raise ValueError(f"cond_metric should be 1 or 2, received {cond_metric} instead.")
    if ax is None:
        fig, ax = plt.subplots()

    ax.semilogy(C,label=label,color=color)

    for x in np.cumsum(next_stage):
        ax.axvline(x, color=color, linestyle='--', linewidth=0.8)
    ax.set_xlabel('nonlinear iterations')
    ax.set_ylabel(f'l{cond_metric} - condtion number')
    # plt.show()
    return C


def plot_Jac_spectrum(db_file, filename=None, max_matrix_size=1000, ax=None,
                      interval=500, repeat=True, color='k', marker='o', s=12):
    pseudo_steps = [int(i.strip('ts_')) for i in list(db_file.keys())]
    pseudo_steps.sort()

    frames = []
    for pseudo_stage in pseudo_steps:
        nl_steps = [int(i.strip('nl_')) for i in list(db_file[f'ts_{pseudo_stage}'].keys())]
        nl_steps.sort()
        for nl_stage in nl_steps[1:]:
            frames.append((pseudo_stage, nl_stage))

    if len(frames) == 0:
        raise ValueError("No Jacobian data found to plot.")

    spectra = []
    for pseudo_stage, nl_stage in frames:
        A = read_free_jacobian_coo(db_file, pseudo_stage, nl_stage)
        if max(A.shape) > max_matrix_size:
            raise ValueError(
                "Jacobian spectrum plotting uses a dense eigenvalue solve and is "
                f"limited to matrices with dimension <= {max_matrix_size}; got {A.shape} "
                f"for pseudo time step {pseudo_stage}, nonlinear iteration {nl_stage}."
            )
        eigvals = np.linalg.eigvals(np.asarray(A.todense()))
        spectra.append((pseudo_stage, nl_stage, eigvals))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    all_eigvals = np.concatenate([eigvals for _, _, eigvals in spectra])
    real_min, real_max = np.min(all_eigvals.real), np.max(all_eigvals.real)
    imag_min, imag_max = np.min(all_eigvals.imag), np.max(all_eigvals.imag)
    real_pad = 0.05 * (real_max - real_min) if real_max > real_min else 1.0
    imag_pad = 0.05 * (imag_max - imag_min) if imag_max > imag_min else 1.0

    scatter = ax.scatter([], [], color=color, marker=marker, s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(0.0, color='0.7', linewidth=0.8)
    ax.axvline(0.0, color='0.7', linewidth=0.8)
    ax.set_xlim(real_min - real_pad, real_max + real_pad)
    ax.set_ylim(0.01, imag_max + imag_pad)
    # ax.set_ylim(imag_min - imag_pad, imag_max + imag_pad)
    ax.set_xlabel('real')
    ax.set_ylabel('imaginary')
    # ax.set_aspect('equal', adjustable='box')
    ax.set_aspect('auto')

    def update(frame):
        pseudo_stage, nl_stage, eigvals = spectra[frame]
        scatter.set_offsets(np.column_stack((eigvals.real, eigvals.imag)))
        ax.set_title(
            f'Jacobian spectrum: pseudo time step {pseudo_stage}, '
            f'nonlinear iteration {nl_stage}'
        )
        return scatter,

    ani = animation.FuncAnimation(
        fig, update, frames=len(spectra), interval=interval, repeat=repeat
    )
    update(0)
    if filename is not None:
        if not animation.writers.is_available('ffmpeg'):
            raise RuntimeError("The matplotlib ffmpeg writer is not available.")
        writer = animation.FFMpegWriter(fps=1000.0 / interval)
        ani.save(filename, writer=writer)
    return ani
