import matplotlib.pyplot as plt
import os



Fibers = [1, 2, 4, 9, 16, 23,36,46]


def get_stats(filepath):
    dofs = None
    solver_time = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Number of dofs:"):
                # "Number of dofs: 1138 (1054 free)"
                dofs = int(line.split(":")[1].split("(")[0].strip())
            elif line.strip().startswith("Total Solver time:"):
                # "	Total Solver time: 21.866555687 seconds"
                solver_time = float(line.split(":")[1].replace("seconds", "").strip())
    return dofs, solver_time

IGFEM_DOFS, IGFEM_time = [],[]
CG_DOFS, CG_time = [],[]
dense_DOFS, dense_time = [],[]

for num_fiber in Fibers:
    filepath_IGFEM = f"tests/IGFEM_ref/{num_fiber}fib_t500/statistics.txt"
    filepath_CG = f"tests/output/nonlinear_IGFEM_vmap_t500_{num_fiber}fib_CG/statistics.txt"
    filepath_dense = f"tests/output/nonlinear_IGFEM_vmap_t500_{num_fiber}fib_dense/statistics.txt"

    try:
        dofs, solver_time = get_stats(filepath_IGFEM)
        IGFEM_DOFS.append(dofs)
        IGFEM_time.append(solver_time)
    except:
        pass


    try:
        dofs, solver_time = get_stats(filepath_CG)
        CG_DOFS.append(dofs)
        CG_time.append(solver_time)
    except:
        pass

    try:
        dofs, solver_time = get_stats(filepath_dense)
        dense_DOFS.append(dofs)
        dense_time.append(solver_time)
    except:
        pass


plt.rcParams['font.size'] = 14

plt.figure(figsize=(8, 6))
plt.plot(IGFEM_DOFS, IGFEM_time, 'o-', color='#4DA6FF', linewidth=2.5, markersize=8, label='IGFEM')
plt.plot(CG_DOFS, CG_time, 's-', color='#2CA02C', linewidth=2.5, markersize=8, label='JAX-FEM (CG)')
plt.plot(dense_DOFS, dense_time, '^-', color='#F08080', linewidth=2.5, markersize=8, label='JAX-FEM (Dense)')

plt.xlabel('Degrees of Freedom (DOFs)')
plt.ylabel('Time (seconds)')
plt.title('Computation Time vs. Degrees of Freedom (DOFs)')
# plt.xticks(DOFS)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('dofs_vs_time.png', dpi=300)