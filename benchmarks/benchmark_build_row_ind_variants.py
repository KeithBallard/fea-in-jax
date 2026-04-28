from helper import *

import meshio
import jax.numpy as jnp
import numpy as np
from numba import njit

from collections import defaultdict

def build_row_ind_v1(n_vertices: jnp.integer, cells: jnp.ndarray) -> jnp.ndarray:
    csr_row_ind = np.zeros((n_vertices + 1,), dtype=np.uint64)
    one = cells.dtype.type(1)
    for cell_index in range(cells.shape[0]):
        csr_row_ind[cells[cell_index] + one] += 1
    return jnp.cumsum(csr_row_ind)


def build_row_ind_v2(n_vertices: int, cells: list) -> jnp.ndarray:
    csr_row_ind = np.zeros((n_vertices + 1,), dtype=np.uint64)
    one = np.uint64(1)
    for cell in cells:
        for vert_index in cell:
            csr_row_ind[np.uint64(vert_index) + one] += 1
    return jnp.cumsum(csr_row_ind)


def build_row_ind_v3(n_vertices: jnp.integer, cells: jnp.ndarray) -> jnp.ndarray:
    # Note: slightly wrong, but fine for benchmarking
    values, counts = np.unique(cells, return_counts=True)
    return jnp.cumsum(counts)


@njit
def build_row_ind_v5(n_vertices: int, cells: np.ndarray) -> np.ndarray:
    csr_row_ind = np.zeros((n_vertices + 1,), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            csr_row_ind[np.uint64(cells[i, j]) + one] += 1
    return csr_row_ind


times_by_version = defaultdict(list)

for i, mesh_size in enumerate([0.005]): #0.05, 0.01, 0.005]):

    mesh = meshio.read(get_mesh(f"polygon_mesh_{mesh_size}.vtk"))

    n_points = len(mesh.points)
    list_cells = mesh.cells[1].data
    jnp_cells = jnp.array(mesh.cells[1].data, dtype=jnp.uint32)
    np_cells = np.array(mesh.cells[1].data, dtype=np.uint32)

    for j, case in enumerate([
        (build_row_ind_v1, n_points, list_cells),
        (build_row_ind_v2, n_points, jnp_cells),
        (build_row_ind_v3, n_points, list_cells),
        (build_row_ind_v5, n_points, np_cells),
    ]):
        n_calls = 40
        f, n, cells = case
        _, times, jit_time, first_call_time = timeit(
            f,
            fixed_kwargs={"n_vertices": n, "cells": cells},
            generated_kwargs={},
            time_jit=True,
            n_calls=n_calls,
            return_timing=True,
        )
        times_by_version[f.__name__].append((mesh_size, times))

import matplotlib.pyplot as plt

for version, time_data in times_by_version.items():
    mesh_sizes = [s for s, t in time_data for i in range(len(t))]
    times = [i for s, t in time_data for i in t]
    plt.plot(mesh_sizes, times, label=version)
plt.legend()
plt.show()