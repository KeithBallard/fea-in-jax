import meshio
import numpy as np
from . import contact
import h5py
import re

import json
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

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



def write_simulation_log(filename, log_filename=None, max_array_items=None, **params):
    """
    Write a JSON run log for a simulation whenever an output filename/base is provided.

    Parameters
    ----------
    filename
        Output filename associated with the run. If None, no log is written.
    log_filename
        Optional explicit log filename. Defaults to filename with ".log.json" suffix.
    max_array_items
        Optional cap for array values. None logs complete arrays.
    **params
        Simulation parameters needed for restart/reproduction, e.g. fabric, materials,
        contact_options, solver_options, boundary_conditions, initial_conditions.
    """
    if filename is None:
        return None

    output_path = Path(filename)
    if log_filename is None:
        log_path = output_path.with_suffix(".log.json")
    else:
        log_path = Path(log_filename)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "output_filename": str(output_path),
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "git": _git_info(),
        "parameters": _to_jsonable(params, max_array_items=max_array_items),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)

    return str(log_path)


def _git_info():
    def git_stdout(args):
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git_stdout(["rev-parse", "HEAD"])
    if commit is None:
        return None

    status = git_stdout(["status", "--short"])
    return {
        "commit": commit,
        "branch": git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _to_jsonable(value, max_array_items=None):
    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return str(value)

    if isinstance(value, np.generic):
        return _to_jsonable(value.item(), max_array_items=max_array_items)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "name": value.name,
            "value": value.value,
        }

    if isinstance(value, slice):
        return {
            "__type__": "slice",
            "start": _to_jsonable(value.start, max_array_items=max_array_items),
            "stop": _to_jsonable(value.stop, max_array_items=max_array_items),
            "step": _to_jsonable(value.step, max_array_items=max_array_items),
        }

    if _is_array_like(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return _to_jsonable(arr.item(), max_array_items=max_array_items)

        out = {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        if max_array_items is not None and arr.size > max_array_items:
            out["truncated"] = True
            out["values_preview"] = _to_jsonable(
                arr.reshape(-1)[:max_array_items].tolist(),
                max_array_items=max_array_items,
            )
        else:
            out["values"] = _to_jsonable(arr.tolist(), max_array_items=max_array_items)
        return out

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            **{
                field.name: _to_jsonable(
                    getattr(value, field.name),
                    max_array_items=max_array_items,
                )
                for field in fields(value)
            },
        }

    if isinstance(value, Mapping):
        return {
            str(_to_jsonable(k, max_array_items=max_array_items)): _to_jsonable(
                v,
                max_array_items=max_array_items,
            )
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v, max_array_items=max_array_items) for v in value]

    if hasattr(value, "func") and hasattr(value, "args"):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "func": _qualified_name(value.func),
            "args": _to_jsonable(value.args, max_array_items=max_array_items),
            "keywords": _to_jsonable(
                getattr(value, "keywords", None),
                max_array_items=max_array_items,
            ),
        }

    if callable(value):
        return {"__callable__": _qualified_name(value)}

    return repr(value)


def _is_array_like(value):
    return isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "__array__")
    )


def _qualified_name(value):
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(value)


