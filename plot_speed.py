import matplotlib.pyplot as plt
import os



Fibers = [1, 2, 4, 9, 16, 23, 33, 36, 46, 49, 60, 64, 100]


def get_stats(filepath):
    dofs = None
    solver_time = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Number of dofs:"):
                dofs = int(line.split(":")[1].split("(")[0].strip())
            elif line.strip().startswith("Total Solver time:"):
                solver_time = float(line.split(":")[1].replace("seconds", "").strip())
    return dofs, solver_time

IGFEM_DOFS, IGFEM_time = [],[]
CG_DOFS, CG_time = [],[]
jetsci_DOFS, jetsci_time = [],[]
scan_DOFS, scan_time = [],[]

for num_fiber in Fibers:
    filepath_IGFEM = f"tests/IGFEM_ref/{num_fiber}fib_t500/statistics.txt"
    filepath_CG = f"tests/output/CG_t500_{num_fiber}fib/statistics.txt"
    filepath_jetsci = f"tests/output/jetsci_t500_{num_fiber}fib/statistics.txt"
    filepath_scan = f"tests/output/SP_t500_{num_fiber}fib/statistics.txt"

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
        dofs, solver_time = get_stats(filepath_jetsci)
        jetsci_DOFS.append(dofs)
        jetsci_time.append(solver_time)
    except:
        pass

    try:
        dofs, solver_time = get_stats(filepath_scan)
        scan_DOFS.append(dofs)
        scan_time.append(solver_time)
    except:
        pass

plt.rcParams['font.size'] = 14

plt.figure(figsize=(8, 6))
plt.plot(IGFEM_DOFS, IGFEM_time, 'o-', color='#4DA6FF', linewidth=2.5, markersize=8, label='IGFEM')
plt.plot(CG_DOFS, CG_time, 's-', color='#2CA02C', linewidth=2.5, markersize=8, label='JAX (CG)')
plt.plot(jetsci_DOFS, jetsci_time, '^-', color='#F08080', linewidth=2.5, markersize=8, label='JetSCI (CG)')
plt.plot(scan_DOFS, scan_time, '^-', color="#A86AF4", linewidth=2.5, markersize=8, label='JAX (SP)')

plt.xlabel('Degrees of Freedom (DOFs)')
plt.ylabel('Time (seconds)')
plt.title('Computation Time vs. Degrees of Freedom (DOFs)')
# plt.xticks(DOFS)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('dofs_vs_time.png', dpi=300)