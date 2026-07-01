from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np

f = read_fabric("initial_single_fiber.fab")
write_vtk(f,get_output(filename = "FabricExample/AngleInterlockTextile.vtk"))
