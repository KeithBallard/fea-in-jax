from fe_jax.helper import *

deb = DebugInfo(
    flags=[
        (DebugOutputQuantities.ELEMENT_JACOBIAN, DebugOutputStage.TIME_STEP),
        (DebugOutputQuantities.ELEMENT_JACOBIAN, DebugOutputStage.NONLINEAR_SOLVE),
        (DebugOutputQuantities.ELEMENT_JACOBIAN, DebugOutputStage.LINEAR_SOLVE),
    ],
    file=h5py.File("test.h5", "w"),
)

# TODO put in JIT section
deb.begin_stage(
    time_step=0,
    nonlinear_solve=0,
    linear_solve=0,
    current_stage=DebugOutputStage.TIME_STEP,
)

assert deb.contains(DebugOutputQuantities.ELEMENT_JACOBIAN) == True

# Simulate two batches
for i in range (2):
    deb.batch_output(
        quantity=DebugOutputQuantities.ELEMENT_JACOBIAN, i=i, arr=i * jnp.ones((2, 2))
    )


