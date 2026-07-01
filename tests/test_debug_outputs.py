from fe_jax.helper import *
jax.config.update("jax_disable_jit", True)

deb = DebugInfo(
    flags=[
        (DebugOutputQuantities.ELEMENT_JACOBIAN, DebugOutputStage.TIME_STEP),
        (DebugOutputQuantities.ELEMENT_JACOBIAN, DebugOutputStage.NONLINEAR_SOLVE),
        (DebugOutputQuantities.ELEMENT_JACOBIAN, DebugOutputStage.LINEAR_SOLVE),
        (DebugOutputQuantities.ELEMENT_RESIDUAL, DebugOutputStage.TIME_STEP),
        (DebugOutputQuantities.ELEMENT_RESIDUAL, DebugOutputStage.NONLINEAR_SOLVE),
        (DebugOutputQuantities.ELEMENT_RESIDUAL, DebugOutputStage.LINEAR_SOLVE),
    ],
    file=h5py.File("test.h5", "w"),
)

# TODO put in JIT section
deb.begin_stage(
    time_step=0,
    nonlinear_solve=0,
    linear_solve=0,
    current_stage=DebugOutputStage.LINEAR_SOLVE,
)

jax.effects_barrier()

assert deb.contains(DebugOutputQuantities.ELEMENT_JACOBIAN) == True
assert deb.contains(DebugOutputQuantities.ELEMENT_RESIDUAL) == True

# Simulate two batches
for i in range (2):
    deb.output(
        quantity=DebugOutputQuantities.ELEMENT_JACOBIAN, name=f"b_{i}", arr=i * jnp.ones((2, 2))
    )
    deb.output(
        quantity=DebugOutputQuantities.ELEMENT_RESIDUAL, name=f"b_{i}", arr=(-i-1) * jnp.ones((2, 2))
    )


