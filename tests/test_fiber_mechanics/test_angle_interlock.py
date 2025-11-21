import sys, os

# Adding parent directory for helper
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from helper import *
from fe_jax.fiber_mechanics import *
from fe_jax.fiber_mechanics.vtk import *

fabric = read_fabric(get_fabric('angle_interlock'))
assert fabric.get_n_bundles() == 128
for i in range(fabric.material_ids.shape[0]):
    assert fabric.bundle_offsets[i + 1] - fabric.bundle_offsets[i] == 10

write_vtk(fabric, get_output('angle_interlock.vtk'))