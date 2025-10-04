import jax.numpy as jnp
import jax
import numpy as np
from functools import partial


with jax.log_compiles(True):

    @jax.jit
    def foo(A: list[jnp.array], B: int) -> jnp.array:
        return jnp.array([jnp.power(a, B) for a in A])
    
    @jax.jit
    def foo2(A: list[jnp.array]) -> jnp.array:
        return jnp.array([jnp.power(a, 2) for a in A])

    # First call triggers a recompilation
    A = [jnp.array(np.random.random((2, 1))) for i in range(2)]
    sqrt_A = foo(A, 1)
    print(f'A = {A}')
    print('\nFinished call 1\n')

    # Does NOT trigger a recompilation since the sizes of the inner arrays are the same
    A = [jnp.array(np.random.random((2, 1))) for i in range(2)]
    sqrt_A = foo(A, 1)
    print(f'A = {A}')
    print('\nFinished call 2\n')

    # Triggers a recompilation since the sizes of the inner arrays are different
    A = [jnp.array(np.random.random((2, 2))) for i in range(2)]
    sqrt_A = foo(A, 1)
    print(f'A = {A}')
    print('\nFinished call 3\n')

    # Throws an errors since the inner lists are different sizes
    A = [jnp.array(np.random.random((2, 2))), jnp.array(np.random.random((2, 1)))]
    try:
        sqrt_A = foo2(A)
    except ValueError:
        print(f'Expected exception raised for A = {A}')
    except Exception:
        raise Exception('Unexpected exception raised')
    else:
        raise Exception('Expected exception not raised')
    
    print('\nFinished call 4\n')