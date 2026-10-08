import itertools
import os

import matplotlib.pyplot as plt


FIBERS = [1, 2, 4, 9, 16, 23, 33, 36, 46, 49, 60, 64, 100]

# One entry per curve. `path` is a template where {n} is replaced by the fiber count.
# `fibers` is optional and defaults to FIBERS.
SERIES = [
    dict(label='IGFEM',       path='tests/IGFEM_ref/{n}fib_t500/statistics.txt'),
    dict(label='JAX (CG)',    path='tests/output/CG_t500_{n}fib/statistics.txt'),
    dict(label='JAX (SP)',    path='tests/output/SP_t500_{n}fib/statistics.txt'),
    dict(label='JAX (update)',    path='tests/output/update_CG_t500_{n}fib/statistics.txt'),
    dict(label='JetSCI (CG)', path='tests/output/jetsci_t500_{n}fib/statistics.txt'),
]

COLORS = ['#4DA6FF', '#2CA02C', '#F08080', '#A86AF4', '#FF7F0E', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']


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


def collect(path_template, fibers):
    dofs_list, time_list = [], []
    for n in fibers:
        filepath = path_template.format(n=n)
        if not os.path.isfile(filepath):
            continue
        dofs, solver_time = get_stats(filepath)
        if dofs is None or solver_time is None:
            print(f"Warning: missing dofs or solver time in {filepath}, skipping")
            continue
        dofs_list.append(dofs)
        time_list.append(solver_time)
    return dofs_list, time_list


plt.rcParams['font.size'] = 14
plt.figure(figsize=(8, 6))

for series, color, marker in zip(SERIES, itertools.cycle(COLORS), itertools.cycle(MARKERS)):
    dofs, times = collect(series['path'], series.get('fibers', FIBERS))
    if not dofs:
        print(f"Warning: no data found for '{series['label']}' ({series['path']})")
        continue
    plt.plot(dofs, times, marker + '-', color=color, linewidth=2.5, markersize=8, label=series['label'])

plt.xlabel('Degrees of Freedom (DOFs)')
plt.ylabel('Time (seconds)')
plt.title('Computation Time vs. Degrees of Freedom (DOFs)')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('dofs_vs_time.png', dpi=300)
