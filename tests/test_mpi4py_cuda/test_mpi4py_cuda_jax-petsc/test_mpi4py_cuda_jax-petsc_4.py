from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = np.arange(10, dtype=np.float64)

if rank == 0:
    comm.Send([data, MPI.DOUBLE], dest=1, tag=0)
elif rank == 1:
    recv = np.empty(10, dtype=np.float64)
    comm.Recv([recv, MPI.DOUBLE], source=0, tag=0)