from enum import Enum, auto
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
import h5py

class DebugOutputQuantities(Enum):
    NODE_RESIDUAL = auto()
    NODE_SOLUTION = auto()
    GLOBAL_JACOBIAN_COO = auto()
    ELEMENT_JACOBIAN = auto()
    ELEMENT_RESIDUAL = auto()
    QUAD_ISVS = auto()


class DebugOutputStage(Enum):
    TIME_STEP = auto()
    NONLINEAR_SOLVE = auto()
    LINEAR_SOLVE = auto()


type DebugFlags = list[tuple[DebugOutputQuantities, DebugOutputStage]]

debug_active_groups: dict[DebugOutputQuantities, h5py.Group] = {}

def _begin_stage(
    flags: DebugFlags,
    file: h5py.File,
    time_step: int,
    nonlinear_solve: int,
    linear_solve: int,
    current_stage: int,
):
    current_stage = DebugOutputStage(int(current_stage))
    match current_stage:
        case DebugOutputStage.TIME_STEP:
            group = file.create_group(f"ts_{time_step}")
        case DebugOutputStage.NONLINEAR_SOLVE:
            group = file.create_group(
                f"ts_{time_step}/nl_{nonlinear_solve}"
            )
        case _:
            group = file.create_group(
                f"ts_{time_step}/nl_{nonlinear_solve}/linear_{linear_solve}"
            )
    for quantity, stage in flags:
        if stage == current_stage:
            debug_active_groups[quantity] = group

def _batch_output(
    quantity: DebugOutputQuantities,
    i: int,
    arr: jnp.ndarray
):
    debug_active_groups[quantity].create_dataset(
        f"{quantity.name.lower()}_batch_{i}",
        data=arr
    )
    return None

@struct.dataclass
class NullDebugInfo:
    def contains(self, quantity: DebugOutputQuantities) -> bool:
        return False

    def begin_stage(self, *args, **kwargs):
        return None

    def batch_output(self, *args, **kwargs):
        return None

NULL_DEBUG_INFO = NullDebugInfo()

@struct.dataclass
class DebugInfo:
    flags: DebugFlags = struct.field(pytree_node = False)
    file: h5py.File   = struct.field(pytree_node = False)

    def contains(self, quantity: DebugOutputQuantities) -> bool:
        return any(f[0] == quantity for f in self.flags)

    @partial(jax.jit, static_argnames=("current_stage",))
    def begin_stage(
        self,
        time_step: int,
        nonlinear_solve: int,
        linear_solve: int,
        current_stage: DebugOutputStage,
    ):
        jax.experimental.io_callback(
            lambda t, n, l, stage: _begin_stage(self.flags, self.file, t, n, l, stage),
            (),
            time_step,
            nonlinear_solve,
            linear_solve,
            jnp.asarray(current_stage.value, dtype = jnp.int32),
            ordered=True,
        )
        # jax.debug.callback(
        #     lambda t, n, l, stage: _begin_stage(self.flags, self.file, t, n, l, stage),
        #     time_step,
        #     nonlinear_solve,
        #     linear_solve,
        #     jnp.asarray(current_stage.value, dtype = jnp.int32),
        #     ordered=True,
        # )

    @partial(jax.jit, static_argnames=("quantity",))
    def batch_output(self, quantity: DebugOutputQuantities, i: int, arr: jnp.ndarray):
        jax.experimental.io_callback(
            lambda i, arr: _batch_output(quantity, i, arr),
            (),
            i,
            arr,
            ordered=False,
        )
