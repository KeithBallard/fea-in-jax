from helper import *

import igl

V, F = uniform_tri_grid(7, 10)
print(f'V {V.shape} {V}')
print(f'F {F.shape} {F}')

igl.write_triangle_mesh(get_output('test_uniform_grid.stl'), V, F)