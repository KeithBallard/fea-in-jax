import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

def get_mesh(mesh_name: str):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "meshes", mesh_name
    )

def get_fabric(fabric_name: str):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "fabrics",
        fabric_name,
        f"{fabric_name}.fab",
    )

def get_output(filename: str):
    rel_path = Path(filename)
    if rel_path.is_absolute():
        raise ValueError(f"filename must be relative to {_REPO_ROOT}, got absolute path: {filename}")
    output_path = _REPO_ROOT / "output" / Path(filename)
    output_path.parent.mkdir(parents = True, exist_ok=True)
    return str(output_path)

def get_debug_output(filename: str):
    rel_path = Path(filename)
    if rel_path.is_absolute():
        raise ValueError(f"filename must be relative to {_REPO_ROOT}, got absolute path: {filename}")
    debug_path = _REPO_ROOT / "debug" / Path(filename)
    debug_path.parent.mkdir(parents = True, exist_ok=True)
    return str(debug_path)

