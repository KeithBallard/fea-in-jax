import os
from itertools import chain
from pathlib import Path

import math
import matplotlib.pyplot as plt
import meshio
import numpy as np
from . import contact

_REPO_ROOT = Path(__file__).resolve().parents[2]

def get_output(
    filename: str,
    subdir: str = ""
):
    output_path = _REPO_ROOT / "output" / subdir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)
    # output_dir = _REPO_ROOT / "output" / subdir
    # output_dir.mkdir(parents = True, exist_ok=True)
    # return str(output_dir/filename)

def write_fabric_mold_contact(
    fabric,
    mold,
    filename,
    contact_params = None,
):
    fabric_cells = np.vstack(
        [
            [
                [j,j+1]
                for j in range(fabric.fiber_offsets[i],fabric.fiber_offsets[i+1]-1)
            ]
            for i in range(fabric.fiber_offsets.shape[0]-1)
        ]
    )

    point_ids = np.concatenate(
        [
            np.full((fabric.fiber_offsets[i + 1] - fabric.fiber_offsets[i],), i)
            for i in range(fabric.fiber_offsets.shape[0] - 1)
        ]
    )
    if mold is not None:
        point_ids = np.concatenate(
            [
                point_ids,
                np.full((mold.points.shape[0],),point_ids.max()+1)
            ]
        )

    cell_ids = np.concatenate(
        [
            np.full((fabric.fiber_offsets[i+1] - fabric.fiber_offsets[i] - 1,), i)
            for i in range(fabric.fiber_offsets.shape[0] - 1)
        ]
    )

    if mold is not None:
        cell_ids = np.concatenate(
            [
                cell_ids,
                np.full((mold.connections.shape[0],),cell_ids.max()+1)
            ]
        )

    if mold is not None:
        points = np.concatenate([fabric.points,mold.points])
        cells = np.concatenate([fabric_cells,mold.connections+fabric.points.shape[0]])
    else:
        points = fabric.points
        cells = fabric_cells
    if contact_params:
        contact_cells = contact.contact_batch(
            points=points,
            point_fiber_ids=point_ids,
            adjacency_block=contact_params['self_adjacency_block'],
            radius=contact_params['contact_search_radius'],
        )
        cells = np.concatenate([cells,contact_cells])
        cell_ids = np.concatenate(
            [
                cell_ids,
                np.full((contact_cells.shape[0],),cell_ids.max()+1)
            ]
        )
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=np.float64),
        cells=[
            ("line", np.asarray(cells, dtype=np.uint64)),
        ],
        point_data={
            "fiber_id": np.asarray(point_ids, dtype=np.int64).reshape(-1),
        },
        cell_data={
            "cell_id": [
                np.asarray(cell_ids, dtype=np.int64).reshape(-1),
            ],
        },
    )

    mesh.write(filename)
