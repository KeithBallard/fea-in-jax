from .basis_quadrature import *

import jax
import jax.numpy as jnp

from flax import struct
from enum import Enum
from typing import Callable, Any


class StrainMeasure(Enum):
    LINEAR = "Linear"
    GREEN_LAGRANGE = "GreenLagrange"
    ENGINEERING = "Engineering"
    LOGARITHMIC = "Logarithmic"

    @classmethod
    def from_value(cls, value: "StrainMeasure | str") -> "StrainMeasure":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.replace("_", "").replace("-", "").replace(" ", "").lower()
            for strain_measure in cls:
                enum_name = strain_measure.name.replace("_", "").lower()
                enum_value = strain_measure.value.lower()
                if normalized in (enum_name, enum_value):
                    return strain_measure
        raise ValueError(f"Option {value} is not a valid choice for StrainMeasure.")


@struct.dataclass
class ElementBatch:
    """
    Describes a batch of elements. Passed into solve_bvp()
    """

    # Defines the type of finite element formulation (basis functions and quadrature) for the
    # elements in this batch.
    fe_type: FiniteElementType
    # Number of degrees of freedom per basis function (typically also per a node)
    n_dofs_per_basis: int
    # List of vertex indices for each element (refers to list of vertices passed to solve_bvp(),
    # not internal batch numbering)
    connectivity_en: np.ndarray[Any, np.dtype[np.uint64]]
    # A callable constitutive model for the batch of elements, which is passed along to the residual
    # function as an argument to perform the residual calculation.
    constitutive_model: jax.tree_util.Partial
    # Array can 1, 2, or 3 dimensions depending on whether parameters are defined 1) the same
    # for all quad points / elements in the batch [shape should be (M,)], 2) the same for all quad points
    # but varying across elements [shape should be (E, M)], or 3) varying across each quad point /
    # element in the batch [shape should be (E, Q, M)], respectively.
    material_params: jnp.ndarray
    # Array can 2 or 3 dimensions depending on whether state variables are defined 1) the same for
    # all quad points but varying across elements [shape should be (E, I)], or 3) varying across
    # each quad point / element in the batch [shape should be (E, Q, I)], respectively.
    internal_state: jnp.ndarray | None = None
    # Strain measure used by residuals/constitutive models that depend on this choice.
    strain_measure: StrainMeasure | str = struct.field(
        default=StrainMeasure.GREEN_LAGRANGE,
        pytree_node=False,
    )

    def __post_init__(self):
        object.__setattr__(
            self,
            "strain_measure",
            StrainMeasure.from_value(self.strain_measure),
        )
        Q = get_quadrature(fe_type=self.fe_type)[0].shape[0]
        if len(self.material_params.shape) == 2:
            # Dimensions should be (E, M)
            assert (
                self.material_params.shape[0] == self.connectivity_en.shape[0]
            ), f"`material_params` had dimension 2, which means the shape should be (E, M). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `material_params.shape[0]` ({self.material_params.shape[0]})"
        elif len(self.material_params.shape) == 3:
            # Dimensions should be (E, Q, M)
            assert (
                self.material_params.shape[0] == self.connectivity_en.shape[0]
            ), f"`material_params` had dimension 3, which means the shape should be (E, Q, M). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `material_params.shape[0]` ({self.material_params.shape[0]})"
            assert (
                self.material_params.shape[1] == Q
            ), f"`material_params` had dimension 3, which means the shape should be (E, Q, M). However, `fe_type` results in Q = {Q}, which did not match `material_params.shape[1]` ({self.material_params.shape[1]})"
        if self.internal_state is not None:
            if len(self.internal_state.shape) == 2:
                # Dimensions should be (E, I)
                assert (
                    self.internal_state.shape[0] == self.connectivity_en.shape[0]
                ), f"`internal_state` had dimension 2, which means the shape should be (E, I). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `internal_state.shape[0]` ({self.internal_state.shape[0]})"
            elif len(self.internal_state.shape) == 3:
                # Dimensions should be (E, Q, M)
                assert (
                    self.internal_state.shape[0] == self.connectivity_en.shape[0]
                ), f"`internal_state` had dimension 3, which means the shape should be (E, Q, I). However, `connectivity_en.shape[0]` ({self.connectivity_en.shape[0]}) did not match `internal_state.shape[0]` ({self.internal_state.shape[0]})"
                assert (
                    self.internal_state.shape[1] == Q
                ), f"`internal_state` had dimension 3, which means the shape should be (E, Q, I). However, `fe_type` results in Q = {Q}, which did not match `internal_state.shape[1]` ({self.internal_state.shape[1]})"
