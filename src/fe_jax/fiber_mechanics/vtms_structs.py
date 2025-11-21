import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from ..np_types import *


@dataclass
class VTMSBundle:
    # Name of the fiber bundle for debugging purposes, derived from file name
    name: str
    # Number of fiber fibers in the bundle / tow
    n_fibers: int
    # Material ID used to determine the properties and constitutive model for all the fibers
    material_id: int
    # Diameter for all the fibers
    diameter: float
    # List of all points (for all fibers in the bundle)
    points: NPArray_2_DD_float64
    # Offsets into `points` indicating start/end of points for each fiber
    fiber_offsets: NPArray_1_D_uint64

    def get_n_bundles(self) -> int:
        # Exists only for compatibility to interop with VTMSFabric
        return 1
    
    def get_material_id(self, bundle_i: int) -> int:
        # Exists only for compatibility to interop with VTMSFabric
        assert bundle_i == 0
        return self.material_id
    
    def get_diameter(self, bundle_i: int) -> float:
        # Exists only for compatibility to interop with VTMSFabric
        assert bundle_i == 0
        return self.diameter
    
    def get_n_fibers_in_bundle(self, bundle_i: int) -> int:
        # Exists only for compatibility to interop with VTMSFabric
        assert bundle_i == 0
        return self.n_fibers
    
    def get_fiber_points(self, bundle_i: int, fiber_i: int):
        # Accepts bundle_i only for compatibility to interop with VTMSFabric
        assert bundle_i == 0
        s = slice(self.fiber_offsets[fiber_i], self.fiber_offsets[fiber_i + 1])
        return self.points[s, :]


@dataclass
class VTMSFabric:
    # Name of the fabric for debugging purposes, derived from file name
    name: str
    # Material ID for each fiber bundle, used to determine the properties and constitutive model
    material_ids: NPArray_1_D_uint64
    # Diameter for each fiber bundle
    diameters: NPArray_1_D_float64
    # List of all points (for all fibers in all fiber bundles)
    points: NPArray_2_DD_float64
    # Offsets into `points` indicating start/end of points for each fiber
    fiber_offsets: NPArray_1_D_uint64
    # Offsets into `fiber_offsets` indicating start/end of each fiber bundle
    bundle_offsets: NPArray_1_D_uint64

    def __post_init__(self):
        assert self.get_n_bundles() == self.diameters.shape[0]
        assert self.get_n_bundles() == self.material_ids.shape[0]
        assert self.fiber_offsets[-1] == self.points.shape[0]
        assert self.bundle_offsets[-1] == self.fiber_offsets.shape[0] - 1

    def get_n_bundles(self) -> int:
        return self.bundle_offsets.shape[0] - 1
    
    def get_material_id(self, bundle_i: int) -> int:
        return self.material_ids[bundle_i]
    
    def get_diameter(self, bundle_i: int) -> float:
        return self.diameters[bundle_i]
    
    def get_n_fibers_in_bundle(self, bundle_i: int) -> int:
        return self.bundle_offsets[bundle_i + 1] - self.bundle_offsets[bundle_i]

    def get_fiber_points(self, bundle_i: int, fiber_i: int):
        s = slice(
            self.fiber_offsets[self.bundle_offsets[bundle_i] + fiber_i],
            self.fiber_offsets[self.bundle_offsets[bundle_i] + fiber_i + 1],
        )
        return self.points[s, :]


def read_fabric(filepath: str | Path) -> VTMSFabric:
    """
    TODO document
    """
    filepath = Path(filepath)
    assert filepath.is_file(), f"File {filepath} did not exist."

    with open(filepath, "r") as f:
        lines = f.readlines()

    start_line = 0
    for i, line in enumerate(lines):
        if "*FABRIC" in line:
            start_line = i + 1
            break

    n_fib_files = int(lines[start_line])
    start_line += 1

    bundles = [read_fib(filepath.parent / line.strip()) for line in lines[start_line:]]
    assert n_fib_files == len(bundles)
    bundle_point_offsets = np.append(
        [0], np.cumsum([b.points.shape[0] for b in bundles])
    )
    return VTMSFabric(
        name=filepath.name,
        material_ids=np.array([b.material_id for b in bundles]),
        diameters=np.array([b.diameter for b in bundles]),
        points=np.vstack([b.points for b in bundles]),
        fiber_offsets=np.hstack(
            [b.fiber_offsets[0:-1] + bundle_point_offsets[i] for i, b in enumerate(bundles)] + [bundle_point_offsets[-1]]
        ),
        bundle_offsets=np.append(
            [0], np.cumsum([b.n_fibers for b in bundles])
        ),
    )


def read_fib(filepath: str | Path) -> VTMSBundle:
    """
    TODO document
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    start_line = 0
    for i, line in enumerate(lines):
        if "#Fiber" in line:
            start_line = i + 1
            break

    tokens = lines[start_line].split()
    n_fibers = int(tokens[0])
    material_id = int(tokens[1])
    diameter = float(tokens[2])
    start_line += 1

    n_points_per_fiber = np.zeros((n_fibers,), dtype=np.integer)

    for i, line in enumerate(lines[start_line : start_line + n_fibers]):
        n_points_per_fiber[i] = int(line.strip().split()[-1])
    start_line += n_fibers

    bundle_points = np.zeros((np.sum(n_points_per_fiber), 3))
    bundle_point_offsets = np.append([0], np.cumsum(n_points_per_fiber))
    for fiber_i in range(n_fibers):
        for i, line in enumerate(
            lines[start_line + 2 : start_line + 2 + n_points_per_fiber[fiber_i]]
        ):
            tokens = line.strip().split()
            bundle_points[bundle_point_offsets[fiber_i] + i] = np.array(
                tuple(map(float, tokens[1:4]))
            )
        start_line += 2 + n_points_per_fiber[fiber_i]

    return VTMSBundle(
        name=filepath.name,
        n_fibers=n_fibers,
        material_id=material_id,
        diameter=diameter,
        points=bundle_points,
        fiber_offsets=bundle_point_offsets,
    )


