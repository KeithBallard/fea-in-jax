"""Plot a JAX-FEM run against its IGFEM reference.

Produces the two figures that used to come from postprocess.ipynb:
  - <feature>_t.png                : strain/stress history of one cell
  - model_comparsion_<feature>.png : IGFEM / JAX-FEM / error contours at a few strains

Edit the SETTINGS block below and run `python plot_comparison.py`.
Any setting can also be overridden on the command line, e.g.
    python plot_comparison.py --case_JAX nonlinear_IGFEM_vmap_t500_16fib_JetSCI --feature s11
"""
# =============================================================================
# SETTINGS
# =============================================================================
# --- What to plot ---
# CASE_JAX = 'CG_t500_49fib'  # run folder under tests/output
CASE_JAX = 'jetsci_t500_60fib'  # run folder under tests/output
# CASE_JAX = 'SP_t500_64fib'  # run folder under tests/output
CASE_IGFEM = None         # folder under tests/IGFEM_ref; None -> '<N>fib_t500' taken from CASE_JAX
FEATURE = 'damage'        # contour field: 'damage', 'e11', 'e22', 'e12', 's11', 's22', 's12'
CELL = 20                 # cell id for the strain/stress history plot
T_END = 500               # last time step to read
NUM_T = 5                 # number of strain levels (rows) in the contour plot
MAX_STRAIN = 0.012        # applied strain at the end of the run (used for the x-axis / row labels)

# --- Contour plot appearance ---
BLACK_FIBERS = False      # damage only: draw fibers black and use a 0-1 damage scale
SHOW_CELL_EDGES = False   # draw mesh lines
CMAP = 'coolwarm'
DAMAGE_RANGE = (-0.2, 1.0)  # colorbar range for damage (other features use IGFEM min/max)
ERROR_SCALE_DAMAGE = 1e-3   # error colorbar max = max(|error|) * this
ERROR_SCALE_OTHER = 1e-2

# --- General style ---
COLOR_IGFEM = '#4DA6FF'
COLOR_JAX = '#F08080'
DPI = 300
WORKERS = None            # parallel VTK readers; None -> min(32, CPU count)
# =============================================================================

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import pyvista as pv

HOME = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HOME, 'tests', 'output')
IGFEM_REF_DIR = os.path.join(HOME, 'tests', 'IGFEM_ref')

HISTORY_FIELDS = ['e11', 'e22', 'e12', 's11', 's22', 's12']


def _read_step(path, cell, feature, keep_feature):
    """Read one VTK file, returning only what the plots need from it."""
    mesh = pv.read(path)
    history = np.array([mesh.cell_data[f][cell] for f in HISTORY_FIELDS])
    snapshot = np.asarray(mesh.cell_data[feature]) if keep_feature else None
    return history, snapshot


def load_run(paths, cell, feature, snapshot_times, workers):
    """Read a VTK series in parallel.

    Returns the HISTORY_FIELDS of `cell` at every step, shape (n_steps, 6), and
    `feature` for all cells at `snapshot_times`, shape (len(snapshot_times), n_cells).
    """
    keep = [t in snapshot_times for t in range(len(paths))]
    n = len(paths)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_read_step, paths, [cell] * n, [feature] * n, keep,
                                chunksize=max(1, n // (4 * workers))))
    history = np.stack([h for h, _ in results])
    snapshots = {t: s for t, (_, s) in enumerate(results) if s is not None}
    return history, np.stack([snapshots[t] for t in snapshot_times])


def cell_polygons(mesh):
    """2D vertex arrays of every cell (mixed triangles/quads)."""
    xy = mesh.points[:, :2]
    return np.split(xy[mesh.cell_connectivity], mesh.offset[1:-1])


def plot_history(strain, hist_JAX, hist_IGFEM, save_path):
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(3, 2, figsize=(8, 9))
    # Strains in the left column, stresses in the right
    for k, name in enumerate(HISTORY_FIELDS):
        a = ax[k % 3, k // 3]
        a.plot(strain, hist_JAX[:, k], '--', color=COLOR_JAX, linewidth=3, label='JAX-FEM', zorder=1)
        a.plot(strain, hist_IGFEM[:, k], color=COLOR_IGFEM, linewidth=3, label='IGFEM', zorder=0)
        a.set_xlabel(r'$\epsilon$ (%)')
        a.set_ylabel(name)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)


def plot_contours(polygons, feature, snap_IGFEM, snap_JAX, strain_labels, save_path,
                  black_fibers=False, show_cell_edges=False):
    error = np.abs(snap_JAX - snap_IGFEM)
    cmap = plt.get_cmap(CMAP).copy()

    if feature == 'damage' and black_fibers:
        # Fibers carry damage = DAMAGE_RANGE[0], which falls below vmin and is drawn black
        fiber_value = DAMAGE_RANGE[0]
        fiber_mask = np.isclose(snap_IGFEM, fiber_value) | np.isclose(snap_JAX, fiber_value)
        cmap.set_under('black')
        vmin, vmax = 0.0, DAMAGE_RANGE[1]
        vmin_e, vmax_e = 0.0, np.max(error[~fiber_mask]) * ERROR_SCALE_DAMAGE
    elif feature == 'damage':
        vmin, vmax = DAMAGE_RANGE
        vmin_e, vmax_e = np.min(error), np.max(error) * ERROR_SCALE_DAMAGE
    else:
        vmin, vmax = np.min(snap_IGFEM), np.max(snap_IGFEM)
        vmin_e, vmax_e = np.min(error), np.max(error) * ERROR_SCALE_OTHER

    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    norm_e = Normalize(vmin=vmin_e, vmax=vmax_e, clip=False)
    cell_edges = ({'edgecolors': 'k'} if show_cell_edges
                  else {'edgecolors': 'none', 'linewidths': 0, 'antialiaseds': False})

    num_t = len(strain_labels)
    fig, ax = plt.subplots(num_t, 3, figsize=(10.5, 3 * num_t), squeeze=False,
                           gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    for i in range(num_t):
        im_feature = ax[i, 0].add_collection(
            PolyCollection(polygons, array=snap_IGFEM[i], cmap=cmap, norm=norm, **cell_edges))
        ax[i, 1].add_collection(
            PolyCollection(polygons, array=snap_JAX[i], cmap=cmap, norm=norm, **cell_edges))
        im_error = ax[i, 2].add_collection(
            PolyCollection(polygons, array=snap_JAX[i] - snap_IGFEM[i], cmap=CMAP, norm=norm_e, **cell_edges))
        ax[i, 0].set_ylabel(r'$\epsilon$ = ' + str(strain_labels[i]) + '%', fontsize=25, rotation=0, labelpad=70)

    cbar = fig.colorbar(im_feature, cax=fig.add_axes([0.85, 0.56, 0.01, 0.3]))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(feature, size=25)
    cbar_e = fig.colorbar(im_error, cax=fig.add_axes([0.85, 0.16, 0.01, 0.3]))
    cbar_e.ax.tick_params(labelsize=20)
    cbar_e.set_label('Error', size=25)

    for a, title in zip(ax[0], ['IGFEM', 'JAX-FEM', 'Error']):
        a.set_title(title, fontsize=22)
    for a in ax.flat:
        a.set_aspect('equal')
        a.set_xticks([])
        a.set_yticks([])
        a.autoscale()
        a.margins(x=0.01, y=0.01)

    fig.subplots_adjust(right=0.8)
    fig.savefig(save_path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)


def plot_run(case_JAX=CASE_JAX, case_IGFEM=CASE_IGFEM, feature=FEATURE, cell=CELL, t_end=T_END,
             num_t=NUM_T, workers=WORKERS, black_fibers=BLACK_FIBERS, show_cell_edges=SHOW_CELL_EDGES):
    if case_IGFEM is None:
        num_fib = re.search(r'_(\d+)fib', case_JAX).group(1)
        case_IGFEM = f'{num_fib}fib_t500'
    workers = workers or min(32, os.cpu_count())

    run_dir = os.path.join(OUTPUT_DIR, case_JAX)
    paths_JAX = [os.path.join(run_dir, 'vtks', f'fea_solve_out_{t}.vtk') for t in range(t_end + 1)]
    paths_IGFEM = [os.path.join(IGFEM_REF_DIR, case_IGFEM, f'History.{t}.vtk') for t in range(t_end + 1)]

    total_time = t_end + 1
    num_t = min(num_t, total_time)
    snapshot_times = [int(t) for t in np.round(np.linspace(0, total_time - 1, num_t))]
    strain_rate = MAX_STRAIN / total_time
    applied_strain = np.round(np.arange(total_time) * strain_rate * 100, 2)

    hist_JAX, snap_JAX = load_run(paths_JAX, cell, feature, snapshot_times, workers)
    hist_IGFEM, snap_IGFEM = load_run(paths_IGFEM, cell, feature, snapshot_times, workers)

    plot_history(applied_strain, hist_JAX, hist_IGFEM, os.path.join(run_dir, 'strain_stress_t.png'))

    polygons = cell_polygons(pv.read(paths_JAX[0]))
    plot_contours(polygons, feature, snap_IGFEM, snap_JAX, applied_strain[snapshot_times],
                  os.path.join(run_dir, f'model_comparsion_{feature}.png'),
                  black_fibers=black_fibers, show_cell_edges=show_cell_edges)
    print(f'Saved plots to {run_dir}')


if __name__ == '__main__':
    # Command-line flags override the SETTINGS block; omitted flags keep its values.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case_JAX', default=CASE_JAX)
    parser.add_argument('--case_IGFEM', default=CASE_IGFEM)
    parser.add_argument('--feature', default=FEATURE)
    parser.add_argument('--cell', type=int, default=CELL)
    parser.add_argument('--t_end', type=int, default=T_END)
    parser.add_argument('--num_t', type=int, default=NUM_T)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--black_fibers', action=argparse.BooleanOptionalAction, default=BLACK_FIBERS)
    parser.add_argument('--show_cell_edges', action=argparse.BooleanOptionalAction, default=SHOW_CELL_EDGES)
    plot_run(**vars(parser.parse_args()))
