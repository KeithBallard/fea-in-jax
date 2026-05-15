from fe_jax.helper import *

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

def test_coo_arrays_sum_duplicates_combines_duplicate_entries():
    A_bcoo = jsparse.random_bcoo(
        key=jax.random.key(seed=100), shape=[500, 500], dtype=jnp.float64, nse=0.2
    )

    A = jsparse.COO(
        (A_bcoo.data, A_bcoo.indices[:, 0], A_bcoo.indices[:, 1]),
        shape=A_bcoo.shape,
        rows_sorted=A_bcoo.indices_sorted,
    )._sort_indices()

    tmp = jsparse.COO(
        (
            jnp.hstack((A.data, A.data)),
            jnp.hstack((A.row, A.row)),
            jnp.hstack((A.col, A.col)),
        ),
        shape=A.shape,
    )._sort_indices()
    A_2x = jsparse.COO(coo_arrays_sum_duplicates(tmp), shape=A.shape)._sort_indices()

    assert jnp.isclose(A_2x.todense(), 2.0 * A.todense()).all()
