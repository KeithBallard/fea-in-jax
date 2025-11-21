import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from fe_jax import *

import meshio
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import chain
import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)

def get_mesh(mesh_name: str):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meshes', mesh_name)

def get_fabric(fabric_name: str):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fabrics', fabric_name, f'{fabric_name}.fab')

def get_output(filename: str):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', filename)