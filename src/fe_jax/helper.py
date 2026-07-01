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
