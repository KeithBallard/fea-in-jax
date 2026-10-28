"""Plot a finished JAX-FEM run against its IGFEM reference.

Runs fe_jax.plot_results.plot_comparison on an existing run folder. New runs of
tests/test_fea_solve_dmg.py call it themselves at the end; use this script to
re-plot older runs or change the plot settings. It produces:
  - strain_stress_t.png            : strain/stress history of one cell
  - model_comparsion_<feature>.png : IGFEM / JAX-FEM / error contours at a few strains
  - model_comparsion_displacement.png : same layout for the displacement magnitude |u|

Edit the SETTINGS block below and run `python plot_comparison.py`.
Any setting can also be overridden on the command line, e.g.
    python plot_comparison.py --case_JAX nonlinear_IGFEM_vmap_t500_16fib_JetSCI --feature s11
"""
# =============================================================================
# SETTINGS
# =============================================================================
# --- What to plot ---
# CASE_JAX = 'update_CG_t500_60fib'  # run folder under tests/output
CASE_JAX = 'jetsci_t500_800fib'  # run folder under tests/output
# CASE_JAX = 'jetsci_ILU_t500_46fib'  # run folder under tests/output
# CASE_JAX = 'SP_t500_1fib'  # run folder under tests/output
CASE_IGFEM = None         # folder under tests/IGFEM_ref; None -> '<N>fib_t500' taken from CASE_JAX
FEATURE = 'damage'        # contour field: 'damage', 'e11', 'e22', 'e12', 's11', 's22', 's12'
CELL = 20                 # cell id for the strain/stress history plot
T_END = 500               # last time step to read
NUM_T = 5                 # number of strain levels (rows) in the contour plot
MAX_STRAIN = 0.012        # applied strain at the end of the run (used for the x-axis / row labels)

# --- Contour plot appearance (colors, colormaps, DPI live in fe_jax/plot_results.py) ---
BLACK_FIBERS = False      # damage only: draw fibers black and use a 0-1 damage scale
SHOW_CELL_EDGES = False   # draw mesh lines
WORKERS = None            # parallel VTK readers; None -> min(32, CPU count)
# =============================================================================

import argparse
import os
import re

import pyvista as pv

from fe_jax.plot_results import plot_comparison

HOME = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HOME, 'tests', 'output')
IGFEM_REF_DIR = os.path.join(HOME, 'tests', 'IGFEM_ref')


def plot_run(case_JAX=CASE_JAX, case_IGFEM=CASE_IGFEM, feature=FEATURE, cell=CELL, t_end=T_END,
             num_t=NUM_T, workers=WORKERS, black_fibers=BLACK_FIBERS, show_cell_edges=SHOW_CELL_EDGES):
    if case_IGFEM is None:
        num_fib = re.search(r'_(\d+)fib', case_JAX).group(1)
        case_IGFEM = f'{num_fib}fib_t500'

    run_dir = os.path.join(OUTPUT_DIR, case_JAX)
    args = {'out_dir': run_dir, 'vtk_dir': os.path.join(run_dir, 'vtks'),
            'ref_dir': os.path.join(IGFEM_REF_DIR, case_IGFEM), 't_total': t_end, 'strain_max': MAX_STRAIN}
    plot_comparison(args, pv.read(os.path.join(args['vtk_dir'], 'fea_solve_out_0.vtk')),
                    feature=feature, cell=cell, num_t=num_t, workers=workers,
                    black_fibers=black_fibers, show_cell_edges=show_cell_edges)


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
