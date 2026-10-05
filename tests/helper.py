import sys
import os
import zipfile

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from fe_jax import *

import meshio
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import chain
import os
from pathlib import Path

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)


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
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/output", exist_ok=True)
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", filename)


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=folder_path)
                zipf.write(abs_path, arcname=rel_path)