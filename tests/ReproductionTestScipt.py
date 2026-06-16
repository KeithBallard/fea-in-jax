
import subprocess
import sys

import scipy.io as sio
import time
import re

file = [f"microscale_2D_r0.vtk"] #,f"microscale_2D_r1.vtk",f"microscale_2D_r2.vtk"]
solver = ["CG_JAX_SCIPY_W_INFO","GMRES_JAX_SCIPY","BICGSTAB_JAX_SCIPY","PETSC"]
PETScSolver = ["0","1","2"]
preconditioner = ["JACOBI","ILU"]

keyword = "full took"
pattern = rf"{keyword}\s*(\d+\.?\d*)"
fileName = "f"
fileNum = 0
dicts = []


outputDict = {"solver":solver,"PETScSolver":PETScSolver,"preconditioner":preconditioner}
for i in file: 
    startTime = time.time()
    resultString = subprocess.run([sys.executable, "tests/test_microscale_bvp.py",i,solver[3],PETScSolver[0],preconditioner[1]],capture_output=True,text=True).stdout
    print("Full process took",time.time()-startTime)
    match = re.search(pattern, resultString)
    solverTime = match.group(1)
    outputDict[fileName + str(fileNum)] = solverTime
    fileNum = fileNum + 1

#dicts.append(outputDict)

#sio.savemat("testOutput.mat",{"testData":dicts})

