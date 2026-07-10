from enum import Enum, auto
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
import h5py

from .paths import get_debug_output

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

_active_groups: dict[DebugOutputQuantities, h5py.Group] = {}
_active_stage: DebugOutputStage | None = None

def _stage_group_name(
    time_step: int,
    nonlinear_solve: int,
    linear_solve: int,
    current_stage: DebugOutputStage,
) -> str:
    match current_stage:
        case DebugOutputStage.TIME_STEP:
            return f"ts_{time_step}"
        case DebugOutputStage.NONLINEAR_SOLVE:
            return  f"ts_{time_step}/nl_{nonlinear_solve}"
        case DebugOutputStage.LINEAR_SOLVE:
            # raise NotImplementedError("LINEAR_SOLVE debug staging is not implement yet.")
            return f"ts_{time_step}/nl_{nonlinear_solve}/linear_{linear_solve}"
        case _:
            raise ValueError(f"Unknown debug output stage: {current_stage!r}")

def _begin_stage(
    flags: DebugFlags,
    file: h5py.File,
    time_step: int,
    nonlinear_solve: int,
    linear_solve: int,
    current_stage: int,
):
    global _active_stage
    _active_groups.clear()
    _active_stage = DebugOutputStage(int(current_stage))

    if not any(stage == _active_stage for _, stage in flags):
        return

    group = file.require_group(_stage_group_name(time_step, nonlinear_solve, linear_solve, _active_stage))
    for quantity, stage in flags:
        if stage == _active_stage:
            _active_groups[quantity] = group

def _quantity_group(
    quantity: DebugOutputQuantities,
    stage_group: h5py.Group
) -> h5py.Group:
    return stage_group.require_group(quantity.name)

def _write_dataset(
    quantity: DebugOutputQuantities,
    stage_group: h5py.Group,
    name: str,
    value: jnp.ndarray,
):
    qgroup = _quantity_group(quantity, stage_group)
    qgroup.create_dataset(name, data=value)
    return None

def _output(
    quantity: DebugOutputQuantities,
    name: str,
    arr: jnp.ndarray,
):
    if _active_stage is None:
        return None
    if quantity not in _active_groups:
        return None
        # raise RuntimeError(
        #     f"Debug out quantity {quantity.name} was not activated for the current stage. "
        #     "Did you call begin_stage(), and is this quantity enabled in flags?"
        #     f"\nActive groups: {_active_groups}"
        # )
    return _write_dataset(
        quantity,
        _active_groups[quantity],
        name,
        arr,
    )

@struct.dataclass
class NullDebugInfo:
    def contains(self, quantity: DebugOutputQuantities) -> bool:
        return False

    def stage_enabled(self, current_stage: DebugOutputStage) -> bool:
        return False

    def begin_stage(self, *args, **kwargs):
        return None

    def output(self, *args, **kwargs):
        return None

NULL_DEBUG_INFO = NullDebugInfo()

@struct.dataclass
class DebugInfo:
    flags: DebugFlags = struct.field(pytree_node = False)
    file: h5py.File   = struct.field(pytree_node = False)

    def contains(self, quantity: DebugOutputQuantities) -> bool:
        return any(q == quantity for q,_ in self.flags)

    def stage_enabled(self, current_stage: DebugOutputStage) -> bool:
        return any(stage == current_stage for _, stage in self.flags)

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

    # @partial(jax.jit, static_argnames=("quantity","name"))
    def output(self, quantity: DebugOutputQuantities, name: str, arr: jnp.ndarray):
        jax.experimental.io_callback(
            lambda value: _output(quantity, name, value),
            (),
            arr,
            ordered = True,
        )

def make_debug_info(
    flags: DebugFlags,
    filename: str,
) -> DebugInfo:
    return DebugInfo(flags = flags, file = h5py.File(get_debug_output(filename),"w"))
