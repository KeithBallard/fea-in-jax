import meshio
import numpy as np
from . import contact
import h5py
import re

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
            point_diameters=contact_params.get('point_diameters'),
            search2radius_ratio=contact_params.get('surface_contact_alpha'),
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

def _natural_key(name: str):
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]

def print_h5_tree(obj, prefix="", is_last=True, show_shape=False, show_dtype=False):
    if isinstance(obj, h5py.File):
        items = sorted(obj.items(), key=lambda kv: _natural_key(kv[0]))
        for idx, (key, child) in enumerate(items):
            last_child = idx == len(items) - 1
            print_h5_tree(
                child,
                prefix="",
                is_last=last_child,
                show_shape=show_shape,
                show_dtype=show_dtype,
            )
        return

    name = obj.name.rsplit("/", 1)[-1]

    if isinstance(obj, h5py.Group):
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{name}/")
        child_prefix = prefix + ("    " if is_last else "│   ")
        items = sorted(obj.items(), key=lambda kv: _natural_key(kv[0]))
        for idx, (key, child) in enumerate(items):
            last_child = idx == len(items) - 1
            print_h5_tree(
                child,
                prefix=child_prefix,
                is_last=last_child,
                show_shape=show_shape,
                show_dtype=show_dtype,
            )
    else:
        connector = "└── " if is_last else "├── "
        extras = []
        if show_shape:
            extras.append(f"-- shape={obj.shape}")
        if show_dtype:
            extras.append(f"-- dtype={obj.dtype}")
        suffix = f" {' '.join(extras)}" if extras else ""
        print(f"{prefix}{connector}{name}{suffix}")
