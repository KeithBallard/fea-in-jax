"""Private level-4 JAX/PETSc primitive implementation.

Nothing outside `v10.JaxCallsPETSc` should import from this package
directly. These modules own the JAX primitive registration, callback lowering,
custom JVP/VJP rules, and raw PETSc callback details.
"""
