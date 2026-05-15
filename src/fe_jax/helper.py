import os
from itertools import chain
from pathlib import Path

# Keep CPU device fanout consistent for scripts that import this helper module.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")


from . import * # noga: F401,F403

import math
import matplotlib.pyplot as plt
import meshio
import numpy as np

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

def get_output(filename: str, subdir: str = ""):
    output_dir = _REPO_ROOT / "output" / subdir
    output_dir.mkdir(parents = True, exist_ok=True)
    return str(output_dir/filename)

    os.makedirs(
        os.path.dirname(os.path.realpath(__file__)) + "/output",
        exist_ok=True
    )
    return os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        "output",
        filename
    )
