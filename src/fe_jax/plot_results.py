"""Post-run plots comparing a JAX-FEM run against its IGFEM reference.

plot_comparison() writes three figures into the run's output folder:
  - strain_stress_t.png                : strain/stress history of one cell
  - model_comparsion_<feature>.png     : IGFEM / JAX-FEM / error contours at a few strains
  - model_comparsion_displacement.png  : same layout for the displacement magnitude |u|

It reads the VTK series written during the run, so call it after the timed solve.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
import pyvista as pv

HISTORY_FIELDS = ['e11', 'e22', 'e12', 's11', 's22', 's12']

CMAP = 'coolwarm'
DAMAGE_RANGE = (-0.2, 1.0)  # colorbar range for damage (other features use IGFEM min/max)
CMAP_ERROR = 'viridis'      # sequential colormap for |JAX - IGFEM|, scaled 0 to max
COLOR_IGFEM = '#4DA6FF'
COLOR_JAX = '#F08080'
DPI = 300


def _read_step(path, cell, feature, keep_snapshot):
    """Read one VTK file, returning only what the plots need from it."""
    mesh = pv.read(path)
    history = np.array([mesh.cell_data[f][cell] for f in HISTORY_FIELDS])
    if not keep_snapshot:
        return history, None
    return history, (np.asarray(mesh.cell_data[feature]), np.asarray(mesh.point_data['displacement'][:, :2]))


def load_run(paths, cell, feature, snapshot_times, workers):
    """Read a VTK series in parallel.

    Returns the HISTORY_FIELDS of `cell` at every step, shape (n_steps, 6), and at
    `snapshot_times`: `feature` for all cells, shape (len(snapshot_times), n_cells), and
    the in-plane displacement of all points, shape (len(snapshot_times), n_points, 2).
    """
    keep = [t in snapshot_times for t in range(len(paths))]
    n = len(paths)
    # forkserver, not fork: the caller usually has JAX running, and forking a multithreaded
    # process can deadlock
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('forkserver')) as pool:
        results = list(pool.map(_read_step, paths, [cell] * n, [feature] * n, keep,
                                chunksize=max(1, n // (4 * workers))))
    history = np.stack([h for h, _ in results])
    snapshots = {t: s for t, (_, s) in enumerate(results) if s is not None}
    snap_feature = np.stack([snapshots[t][0] for t in snapshot_times])
    snap_u = np.stack([snapshots[t][1] for t in snapshot_times])
    return history, snap_feature, snap_u


def cell_polygons(mesh):
    """2D vertex arrays of every cell (mixed triangles/quads)."""
    xy = mesh.points[:, :2]
    return np.split(xy[mesh.cell_connectivity], mesh.offset[1:-1])


def cell_triangulation(mesh):
    """Fan-triangulate every cell so point data can be drawn with Gouraud shading."""
    triangles = []
    for cell in np.split(mesh.cell_connectivity, mesh.offset[1:-1]):
        triangles += [(cell[0], cell[k], cell[k + 1]) for k in range(1, len(cell) - 1)]
    return Triangulation(mesh.points[:, 0], mesh.points[:, 1], np.array(triangles))


def plot_history(strain, hist_JAX, hist_IGFEM, save_path):
    with matplotlib.rc_context({'font.size': 16}):
        fig = Figure(figsize=(8, 9))
        ax = fig.subplots(3, 2)
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


def plot_contours(polygons, feature, snap_IGFEM, snap_JAX, strain_labels, save_path,
                  black_fibers=False, show_cell_edges=False, triangulation=None, error=None):
    """IGFEM / JAX-FEM / error contours, one row per strain level.

    Values are per cell, unless `triangulation` is given: then they are per point and are
    drawn with Gouraud shading. `error` defaults to |snap_JAX - snap_IGFEM|.
    """
    if error is None:
        error = np.abs(snap_JAX - snap_IGFEM)
    cmap = matplotlib.colormaps[CMAP].copy()

    if feature == 'damage' and black_fibers:
        # Fibers carry damage = DAMAGE_RANGE[0], which falls below vmin and is drawn black
        fiber_value = DAMAGE_RANGE[0]
        fiber_mask = np.isclose(snap_IGFEM, fiber_value) | np.isclose(snap_JAX, fiber_value)
        cmap.set_under('black')
        vmin, vmax = 0.0, DAMAGE_RANGE[1]
        vmax_e = np.max(error[~fiber_mask])
    elif feature == 'damage':
        vmin, vmax = DAMAGE_RANGE
        vmax_e = np.max(error)
    else:
        vmin, vmax = np.min(snap_IGFEM), np.max(snap_IGFEM)
        vmax_e = np.max(error)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    norm_e = Normalize(vmin=0.0, vmax=vmax_e)
    cell_edges = ({'edgecolors': 'k'} if show_cell_edges
                  else {'edgecolors': 'none', 'linewidths': 0, 'antialiaseds': False})

    def draw(a, values, cmap, norm):
        if triangulation is None:
            return a.add_collection(PolyCollection(polygons, array=values, cmap=cmap, norm=norm, **cell_edges))
        im = a.tripcolor(triangulation, values, shading='gouraud', cmap=cmap, norm=norm)
        if show_cell_edges:
            a.add_collection(PolyCollection(polygons, facecolors='none', edgecolors='k', linewidths=0.5))
        return im

    num_t = len(strain_labels)
    fig = Figure(figsize=(10.5, 3 * num_t))
    ax = fig.subplots(num_t, 3, squeeze=False, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    for i in range(num_t):
        im_feature = draw(ax[i, 0], snap_IGFEM[i], cmap, norm)
        draw(ax[i, 1], snap_JAX[i], cmap, norm)
        im_error = draw(ax[i, 2], error[i], CMAP_ERROR, norm_e)
        ax[i, 0].set_ylabel(r'$\epsilon$ = ' + str(strain_labels[i]) + '%', fontsize=25, rotation=0, labelpad=70)

    cbar = fig.colorbar(im_feature, cax=fig.add_axes([0.85, 0.56, 0.01, 0.3]))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(feature, size=25)
    cbar_e = fig.colorbar(im_error, cax=fig.add_axes([0.85, 0.16, 0.01, 0.3]))
    cbar_e.ax.tick_params(labelsize=20)
    cbar_e.set_label('|Error|', size=25)

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


def plot_comparison(args, vtk_mesh, feature='damage', cell=20, num_t=5, workers=None,
                    black_fibers=False, show_cell_edges=False):
    """Plot a finished JAX-FEM run against its IGFEM reference.

    args      : the run's settings; uses 'vtk_dir' (fea_solve_out_{0..t_total}.vtk), 'ref_dir'
                (IGFEM History.{0..t_total}.vtk, plots skipped if missing), 'out_dir' (where
                the PNGs go), 't_total' and 'strain_max'
    vtk_mesh  : the run's pyvista mesh (only its geometry is used)
    feature   : contour field: 'damage', 'e11', 'e22', 'e12', 's11', 's22', 's12'
    cell      : cell id for the strain/stress history plot
    num_t     : number of strain levels (rows) in the contour plots
    workers   : parallel VTK readers; None -> min(32, CPU count)
    """
    t_total, out_dir = args['t_total'], args['out_dir']
    if not os.path.isdir(args['ref_dir']):
        print(f"No IGFEM reference at {args['ref_dir']}, skipping comparison plots")
        return
    workers = workers or min(32, os.cpu_count())

    paths_JAX = [os.path.join(args['vtk_dir'], f'fea_solve_out_{t}.vtk') for t in range(t_total + 1)]
    paths_IGFEM = [os.path.join(args['ref_dir'], f'History.{t}.vtk') for t in range(t_total + 1)]

    num_t = min(num_t, t_total + 1)
    snapshot_times = [int(t) for t in np.round(np.linspace(0, t_total, num_t))]
    # Step t applies strain_max * t / t_total, in percent
    applied_strain = np.arange(t_total + 1) * args['strain_max'] / t_total * 100
    strain_labels = np.round(applied_strain[snapshot_times], 2)

    hist_JAX, snap_JAX, u_JAX = load_run(paths_JAX, cell, feature, snapshot_times, workers)
    hist_IGFEM, snap_IGFEM, u_IGFEM = load_run(paths_IGFEM, cell, feature, snapshot_times, workers)

    plot_history(applied_strain, hist_JAX, hist_IGFEM, os.path.join(out_dir, 'strain_stress_t.png'))
    polygons = cell_polygons(vtk_mesh)
    plot_contours(polygons, feature, snap_IGFEM, snap_JAX, strain_labels,
                  os.path.join(out_dir, f'model_comparsion_{feature}.png'),
                  black_fibers=black_fibers, show_cell_edges=show_cell_edges)
    plot_contours(polygons, '|u|', np.linalg.norm(u_IGFEM, axis=-1), np.linalg.norm(u_JAX, axis=-1),
                  strain_labels, os.path.join(out_dir, 'model_comparsion_displacement.png'),
                  show_cell_edges=show_cell_edges, triangulation=cell_triangulation(vtk_mesh),
                  error=np.linalg.norm(u_JAX - u_IGFEM, axis=-1))
    print(f'Saved plots to {out_dir}')
