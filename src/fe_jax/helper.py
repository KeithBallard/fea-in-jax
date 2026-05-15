import os
from itertools import chain
from pathlib import Path

# Keep CPU device fanout consistent for scripts that import this helper module.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from . import * # noga: F401,F403

import math
import matplotlib.pyplot as plt
import meshio
import numpy as np

# os.environ["XLA_FLAGS"] = ("--xla_force_host_platform_device_count=8"  # Use 8 CPU devices)

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
