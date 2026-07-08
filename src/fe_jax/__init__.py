import jax

jax.config.update("jax_enable_x64", True)

import pathlib
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).parent.resolve() / "__jax_cache__"))

from .np_types import *
from .basis_quadrature import *
from .fea import *
from .linear_elasticity import *
from .hyperelasticity import *
from .profiling import *
from .setup import *
from .sparse_matrix import *
from .utils import *
from .constraint_system import *
from .constraints import *
