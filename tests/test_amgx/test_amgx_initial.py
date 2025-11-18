import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import pyamgx

pyamgx.initialize()

# Initialize config and resources:
cfg = pyamgx.Config().create_from_dict({
        "config_version": 2, 
        "solver": {
            "print_grid_stats": 1, 
            "solver": "AMG", 
            "algorithm":"AGGREGATION",
            "selector": "SIZE_4",
            "print_solve_stats": 1, 
            "smoother": "JACOBI_L1",
            "presweeps": 0, 
            "postsweeps": 3,
            "obtain_timings": 1, 
            "max_iters": 100, 
            "monitor_residual": 1, 
            "convergence": "RELATIVE_INI", 
            "scope": "main", 
            "max_levels": 50, 
            "cycle": "CG", 
            "tolerance" : 1e-06, 
            "norm": "L2" 
        }
    })

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

# Upload system:

M = sparse.csr_matrix(np.random.rand(50, 50))
rhs = np.random.rand(50)
sol = np.zeros(50, dtype=np.float64)

A.upload_CSR(M)
b.upload(rhs)
x.upload(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)

# Download solution
x.download(sol)
print("pyamgx solution: ", sol)
print("scipy solution: ", splinalg.spsolve(M, rhs))

b.upload(rhs)
x.upload(sol)
solver.solve(b, x)

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()