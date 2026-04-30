import jax, jax.numpy as jnp

@jax.jit
def optimizer(x, tol = 1, max_steps = 5):
    
    def cond(arg):
        step, x, history = arg
        # Note that the following would be tempting but breaks jit:
        # return step < max_steps and x > tol
        return (step < max_steps) & (x > tol)
    
    def body(arg):
        step, x, history = arg
        x = x / 2 # simulate taking an optimizer step
        history = history.at[step].set(x) # simulate saving current step
        return (step + 1, x, history)

    return jax.lax.while_loop(
        cond,
        body,
        (0, x, jnp.full(max_steps, jnp.nan))
    )

optimizer(10.) # works