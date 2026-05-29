import os
from itertools import chain
from pathlib import Path

import math
import matplotlib.pyplot as plt
import meshio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

def get_output(filename: str, subdir: str = ""):
    output_path = _REPO_ROOT / "output" / subdir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)
    # output_dir = _REPO_ROOT / "output" / subdir
    # output_dir.mkdir(parents = True, exist_ok=True)
    # return str(output_dir/filename)
