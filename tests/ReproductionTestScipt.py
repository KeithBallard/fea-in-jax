
import subprocess
import sys

import scipy as sc
import time
import re

file = [f"microscale_2D_r0.vtk",f"microscale_2D_r1.vtk",f"microscale_2D_r2.vtk"]
solver = ["CG_JAX_SCIPY_W_INFO","GMRES_JAX_SCIPY","BICGSTAB_JAX_SCIPY","PETSC"]
PETScSolver = ["0","1","2"]
preconditioner = ["JACOBI","ILU"]

keyword = "full took"
pattern = rf"{keyword}\s*(\d+\.?\d*)"

"""
for i in file:
    startTime = time.time()
    resultString = subprocess.run([sys.executable, "tests/test_microscale_bvp.py",file[2],solver[0],PETScSolver[0],preconditioner[0]],capture_output=True,text=True).stdout
    print("Full process took",time.time()-startTime)
    match = re.search(pattern, resultString)
    number = match.group(1)


    


"""

"""startTime = time.time()
resultString = subprocess.run([sys.executable, "tests/test_microscale_bvp.py",file[2],solver[0],PETScSolver[0],preconditioner[0]],capture_output=True,text=True).stdout
endTime1 = time.time() - startTime

"""


startTime = time.time()
resultString = subprocess.run([sys.executable, "tests/test_microscale_bvp.py",file[0],solver[3],PETScSolver[0],preconditioner[0]],capture_output=True,text=True).stdout
endTime2 = time.time() - startTime

