"""PETSc Python Mat context whose multiplication is implemented by JAX."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


class JaxMatContext:
    """PETSc Mat context that forwards `Mat.mult` to a JAX callable.

    This first version copies through host arrays. The purpose is to stabilize
    the KSP -> JAX call shape before optimizing the data boundary.
    """

    def __init__(self, matvec, *, dmplex=None, block_until_ready=True):
        self.matvec = matvec
        self.dmplex = dmplex
        self.local_x = dmplex.createLocalVec() if dmplex is not None else None
        self.block_until_ready = block_until_ready
        self.mult_calls = 0

    def mult(self, mat, x, y):
        """Compute `y = A @ x` by calling the configured JAX matvec."""
        del mat
        self.mult_calls += 1

        x_jax = jnp.asarray(np.asarray(x.getArray(readonly=True)))
        y_jax = self.matvec(x_jax)
        if self.block_until_ready and hasattr(y_jax, "block_until_ready"):
            y_jax.block_until_ready()

        y_array = y.getArray(readonly=False)
        y_array[...] = np.asarray(jax.device_get(y_jax))

    def multTranspose(self, mat, x, y):
        """Compute `y = A.T @ x` if a transpose matvec is installed."""
        transpose_matvec = getattr(self, "transpose_matvec", None)
        if transpose_matvec is None:
            raise NotImplementedError("JaxMatContext does not define transpose matvec")

        del mat
        x_jax = jnp.asarray(np.asarray(x.getArray(readonly=True)))
        y_jax = transpose_matvec(x_jax)
        if self.block_until_ready and hasattr(y_jax, "block_until_ready"):
            y_jax.block_until_ready()

        y_array = y.getArray(readonly=False)
        y_array[...] = np.asarray(jax.device_get(y_jax))
