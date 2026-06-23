from enum import Enum, auto

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

debug_active_groups: dict[DebugOutputQuantities, h5py.Group]


def __begin_stage(
    flags: DebugFlags,
    file: h5py.File,
    time_step: int,
    nonlinear_solve: int,
    linear_solve: int,
    current_stage: DebugOutputStage,
):
    for f in flags:
        if f[1] == current_stage:
            match current_stage:
                case DebugOutputStage.TIME_STEP:
                    debug_active_groups[f[0]] = file.create_group(f"ts_{time_step}")
                case DebugOutputStage.NONLINEAR_SOLVE:
                    debug_active_groups[f[0]] = file.create_group(
                        f"ts_{time_step}/nl_{nonlinear_solve}"
                    )
                case _:
                    debug_active_groups[f[0]] = file.create_group(
                        f"ts_{time_step}/nl_{nonlinear_solve}/linear_{linear_solve}"
                    )


@struct.dataclass
class DebugInfo:
    flags: DebugFlags
    file: h5py.File

    def contains(self, quantity: DebugOutputQuantities) -> bool:
        return any(f[0] == quantity for f in self.flags)

    @jax.jit
    def begin_stage(
        self,
        time_step: int,
        nonlinear_solve: int,
        linear_solve: int,
        current_stage: DebugOutputStage,
    ):
        jax.debug.callback(ordered=True)(
            __begin_stage,
            self.file,
            self.flags,
            time_step,
            nonlinear_solve,
            linear_solve,
            current_stage,
        )

    @jax.jit
    def batch_output(self, quantity: DebugOutputQuantities, i: int, arr: jnp.ndarray):
        jax.debug.callback(ordered=False)(
            lambda i, arr: debug_active_groups[quantity].create_dataset(
                f"element_jacobian_batch_{i}", data=arr
            ),
            i,
            arr,
        )
