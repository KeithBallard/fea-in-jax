v10 PETSc/SNES Differentiation Split
====================================

This version separates the PETSc/JAX prototype into explicit layers and is the
workspace for making PETSc SNES solves differentiable.

Current design question
-----------------------

There are two distinct problems that should stay separate while this version
evolves:

1. PETSc/JAX execution architecture:
   - PETSc owns SNES/KSP/PC iteration and solver state.
   - JAX owns residual, Jacobian, matrix-vector, and vector-output kernels.
   - PETSc callbacks convert PETSc Vec input to JAX arrays through DLPack.
   - JAX outputs are assigned back into PETSc Vec/Mat objects through direct
     DLPack/CuPy paths when PETSc is the outer driver.
   - Matrix callbacks are split into:
     - a JAX evaluation stage that produces `COOData` or a dense JAX matrix,
     - a PETSc assignment stage that chooses rebuild/preallocation or
       values-only update based on the observed sparsity pattern.

2. Autodiff contract:
   - Do not differentiate through SNES iterations.
   - Treat the converged solution `x_star` as defined implicitly by
     `R(args, x_star) = 0`.
   - `args` is the current analogue of the implicit-function parameter `phi`.
   - Differentiation should use the implicit function theorem:

```text
R_x x_dot = -R_args args_dot
R_x.T lambda = x_bar
args_bar = -R_args.T lambda
```

Near-term implication:

- Keep the working PETSc-as-outer-driver SNES callback path as the execution
  target.
- Build a differentiable wrapper around the SNES solve whose public signature
  exposes `args` as a JAX-visible argument.
- Use custom JVP plus a linear-solve primitive when forward-mode Jacobians are
  required.
- Reuse PETSc KSP/transpose KSP solves for the linear and adjoint systems in
  the derivative rules.

Opposite call directions
------------------------

This version now has two separate PETSc/JAX directions:

- `JaxCallsPETSc`: JAX-visible matrix data calls PETSc KSP.
- `PETScCallsJax`: PETSc KSP calls a JAX matvec through a PETSc Python Mat.

Level 2: solver-call workflows
------------------------------

Import from:

```
v10.JaxCallsPETSc
```

This layer batches level-3 method calls into common workflows:

- build a Mat/PC/KSP object bundle
- reuse that bundle for solves or simple solve sequences
- optionally update matrix values for fixed-sparsity systems
- cleanup the whole bundle

Example:

```python
from v10.JaxCallsPETSc import (
    buildSolverObjects,
    solveWithSolverObjects,
    cleanupSolverObjects,
)

solver_objects = buildSolverObjects(matrix_function, x, options)
x = solveWithSolverObjects(solver_objects, b)
cleanupSolverObjects(solver_objects)
```

KSP -> JAX: PETSc Python Mat workflows
--------------------------------------

Import from:

```
v10.PETScCallsJax
```

This layer builds a PETSc Python `Mat` whose `mult` calls a JAX function, then
uses that matrix inside PETSc KSP:

```python
from v10.PETScCallsJax import (
    buildSolverObjects,
    solveWithSolverObjects,
    cleanupSolverObjects,
)

solver_objects = buildSolverObjects(shape, jax_matvec, options)
x = solveWithSolverObjects(solver_objects, b)
cleanupSolverObjects(solver_objects)
```

This is currently a performance/prototyping path and copies through host memory
inside `Mat.mult`; the data boundary can be optimized later. This path defaults
to `PC.NONE` because PETSc does not have an assembled matrix to factor when the
operator is a Python Mat.

SNES nonlinear helpers
----------------------

Import from:

```
v10.NonlinearSNES
```

This package owns SNES-focused conversion helpers for now:

- `convertJAXVecFuncToPETScVecFunc` converts JAX vector-input/vector-output
  functions to PETSc Vec-mutating callbacks.
- `mat_function_converters.py` is the planned home for JAX matrix-output to
  PETSc Mat callbacks, including fixed-pattern and rebuild-on-pattern-change
  paths.

Level 3: PETSc methods and lifetime control
-------------------------------------------

Import from:

```
v10.JaxCallsPETSc
```

This is the layer normal project code should call when it needs manual
PETSc lifecycle control:

- initialize/update/cleanup a PETSc Mat
- initialize/cleanup a PETSc PC
- initialize/cleanup a PETSc KSP
- solve with a KSP/PC and JAX-visible COO matrix values

Level 4: private JAX primitive and differentiation internals
------------------------------------------------------------

Private package:

```
v10.JaxCallsPETSc._primitives
```

This package contains the Keith callback files and primitive rules. It owns:

- `buffer_callback` implementations
- object stores and raw PETSc handles
- primitive registration
- custom JVP/VJP/transpose/batching/lowering rules

Do not import this package from user-facing or nonlinear solver code. If a
level-3 method is missing, add a wrapper in `JaxCallsPETSc` rather than calling
level 4 directly.

The `PETScCallsJax` direction does not have custom primitives yet. Its current
private boundary is the PETSc Python Mat context and the optional JAX
`pure_callback` around `ksp.solve`.

Current intended call shape
---------------------------

```python
from v10.JaxCallsPETSc import (
    PETScMethodOptions,
    init_matrix_from_function,
    init_pc,
    init_ksp,
    KSP_solve,
    cleanup_ksp,
    cleanup_pc,
    cleanup_matrix,
)

matrix = init_matrix_from_function(matrix_function, x, options)
pc = init_pc(matrix, options)
ksp = init_ksp(matrix, options)
A = matrix_function(x)
x = KSP_solve(ksp, pc, A, b)
cleanup_ksp(ksp)
cleanup_pc(pc)
cleanup_matrix(matrix)
```

The level-2 helpers in this same package are built on top of these methods.
The eventual level-1 API should sit above the direction packages.
