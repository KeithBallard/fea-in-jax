import matplotlib.pyplot as plt


# Fibers[       1,     2,   4,    9,    16,   23,     33,    36,    46,    49,    60,    64]
DOFS        = [1138, 1758, 2610, 4746, 7546, 10794, 14802, 16798, 19570, 21830, 27086, 29874]
CG_iter_min = [170,  300,  565,  872,  1167, 967,   1572,  1682,  1718,  1375,  2010,  2153]
CG_iter_max = [318,  505,  729,  1140, 1718, 1301,  2666,  3405,  2695,  1748,  2734,  2734]

plt.rcParams['font.size'] = 14

plt.figure(figsize=(8, 6))
plt.plot(DOFS, CG_iter_min, '^-', linewidth=2.5, markersize=8, label='JAX-FEM (CG) min')
plt.plot(DOFS, CG_iter_max, '^-', linewidth=2.5, markersize=8, label='JAX-FEM (CG) max')

plt.xlabel('Degrees of Freedom (DOFs)')
plt.ylabel('Iterations')
plt.title('Iterations vs. Degrees of Freedom (DOFs)')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('dofs_vs_iterations.png', dpi=300)
