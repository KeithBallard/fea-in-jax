import jax

jax.config.update("jax_enable_x64", True)

import pathlib
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).parent.resolve() / "__jax_cache__"))

from .solve import *
