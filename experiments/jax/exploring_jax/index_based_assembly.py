from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
import fe_jax as fea
from fe_jax.profiling import timeit

from flax import struct
# jax.config.update("jax_platforms","cpu")
"""
It may be possible to use suitable oob behavior of the .at[].* methods to avoid some of the explicit handling of padding, 
especially the where(index==-1,special case,array[index]) constructs. 
The potential was noted when the patchified BCs for unconstrained patches did not break anything despite being "full" of constraints
This script is for examining the performance of those options, to ensure there is no unexpected slowdowns 
(no major speedups are expected either, but would not be unwelcome) as well as determining what replacement is possible.
"""

@struct.dataclass
class AssemblyMap:
    """
    Container for the information required to perform EN-V and V-EN transformations via indexing rather than sparse matmul.

    Fields
    ---------
    indices: Array of shape (EN,) whose entries are the indices from a vertex-based array corresponding to each element-node. 
        Equal to the row_indices array of the sparse array approach, with entries sorted in EN order (such that the col_indices array would be exactly jnp.arange(EN))
    shape: tuple with entries (V,EN), equal to the shape of a sparse array whose matmuls produce the desired transformations
    """

    indices: jnp.ndarray
    shape: tuple[int] = struct.field(pytree_node=False)  

@partial(jax.jit,static_argnames = "n_vertices")
def mesh_to_sparse_assembly_map(
    n_vertices: int,
    cells: jnp.ndarray,
):
    """
    Builds a compressed sparse row (CSR) matrix that serves as a map between two representations
    for vectors on a patch: 1) assembled format and 2) element-node format.

    Takes advantage of the fact that the assembly map has exactly one entry in each column, so the COO representation has a convenient structure.
    """      
    nse = np.prod(cells.shape)
    coo_data = jnp.ones(nse)
    # Search breaks if the padding is over 50%, so ensure the padding is maxed now.
    coo_rowindices = jnp.searchsorted(
        jnp.arange(n_vertices),
        cells.flatten(),
        method="scan_unrolled",
    )
    coo_colindices = jnp.arange(nse)

    return jsparse.BCOO(
        (
            coo_data,
            jnp.stack((coo_rowindices, coo_colindices)).T,
        ),
        shape=(n_vertices, nse),
    )


@partial(jax.jit, static_argnames=["E"])
def sparse_transform_global_to_element_node(
    assembly_map: jsparse.BCOO, v_g: jnp.ndarray, E: int
):
    """
    Transforms a vector that represents a global assembled vector into the element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    U = v_g.shape[1]
    N = assembly_map.shape[1]//E    
    return jsparse.bcoo_dot_general(
        assembly_map,
        v_g,
        dimension_numbers=(((0,), (0,)), ((), ())),
    ).reshape(E, N, U)


@partial(jax.jit, static_argnames=["E"])
def sparse_transform_global_unraveled_to_element_node(
    assembly_map: jsparse.BCOO, v_g: jnp.ndarray, E: int
):
    """
    Transforms a vector that represents a global assembled vector that is unraveled into the
    element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    V = assembly_map.shape[0]
    U = v_g.shape[0] // V
    N = assembly_map.shape[1] // E
    return jsparse.bcoo_dot_general(
        assembly_map,
        v_g.reshape(V, U),
        dimension_numbers=(((0,), (0,)), ((), ())),
    ).reshape(E, N, U)

@jax.jit
def sparse_transform_element_node_to_global_unraveled_nosum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = jsparse.bcoo_dot_general(
        assembly_map,
        jnp.ones(v_en.shape[0] * v_en.shape[1]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    n_cell_per_vert = jnp.where(n_cell_per_vert==0,1,n_cell_per_vert)
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return (v_g / n_cell_per_vert[:, jnp.newaxis]).reshape(
        np.prod(v_g.shape)
    )


@jax.jit
def sparse_transform_element_node_to_global_unraveled_sum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return v_g.reshape(np.prod(v_g.shape))


@jax.jit
def sparse_transform_element_node_to_global_sum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return v_g.reshape(np.prod(v_g.shape) // v_en.shape[2], v_en.shape[2])


@jax.jit
def sparse_transform_element_node_to_global_nosum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = jsparse.bcoo_dot_general(
        assembly_map,
        jnp.ones(v_en.shape[0] * v_en.shape[1]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    v_g = jsparse.bcoo_dot_general(
        assembly_map,
        v_en.reshape(v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    return (v_g / n_cell_per_vert[:, jnp.newaxis]).reshape(np.prod(v_g.shape)//v_en.shape[2],v_en.shape[2])




"""
Straight indexing approach
"""


@partial(jax.jit, static_argnames="n_vertices")
def mesh_to_index_assembly_map(
    n_vertices: int,
    cells: jnp.ndarray,
):
    """
    Generates an array of indices to convert between vertex-labeled values and element-node-labeled values
    """
    coo_rowindices = jnp.searchsorted(
        jnp.arange(n_vertices),
        cells,
        method="scan_unrolled",
    )
    return AssemblyMap(indices=coo_rowindices, shape=(n_vertices, np.prod(cells.shape)))


@jax.jit
def index_transform_global_to_element_node(
    assembly_map: AssemblyMap, v_g: jnp.ndarray
):
    """
    Transforms a vector that represents a global assembled vector into the element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    return v_g.at[assembly_map.indices, :].get(mode="drop", fill_value=0)


@jax.jit
def index_transform_global_unraveled_to_element_node(
    assembly_map: AssemblyMap, v_g: jnp.ndarray
):
    """
    Transforms a vector that represents a global assembled vector that is unraveled into the
    element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    V = assembly_map.shape[0]
    U = v_g.shape[0] // V
    return (
        v_g.reshape((V, U)).at[assembly_map.indices, :].get(mode="drop", fill_value=0)
    )


@jax.jit
def index_transform_element_node_to_global_unraveled_nosum(
    assembly_map: AssemblyMap, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.zeros((V, U)).at[assembly_map.indices, ...].set(v_en, mode="drop")
    return v_g.reshape(np.prod(v_g.shape))


@jax.jit
def index_transform_element_node_to_global_unraveled_sum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.zeros((V, U)).at[assembly_map.indices, ...].add(v_en, mode="drop")
    return v_g.flatten()


@jax.jit
def index_transform_element_node_to_global_sum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.empty((V, U)).at[assembly_map.indices, ...].add(v_en, mode="drop")
    return v_g


@jax.jit
def index_transform_element_node_to_global_nosum(
    assembly_map: jsparse.BCOO, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.zeros((V, U)).at[assembly_map.indices, ...].set(v_en, mode="drop")
    return v_g



def benchmark_comparator(
    funcdict: dict, assembly_maps=None, assert_equal=True, **kwargs
):
    retvals = []
    for name, func in funcdict.items():
        print(name)
        if assembly_maps is not None:
            kwargs.update({"assembly_map": assembly_maps[name]})
        retval = timeit(func, n_calls=N, time_jit=time_jit, **kwargs)[0]
        retvals.append(retval)
    if assert_equal:
        assert all(jnp.allclose(retvals[0], retval) for retval in retvals[1:])
        return retvals[0]
    else:
        return retvals

if __name__ == "__main__":
    N = 10
    N_jacs = 10
    time_jit = True
    derivatives = True
    n=100

    vertices_vd = np.vstack(
        [
            np.ravel(vec)
            for vec in np.meshgrid(
                np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
            )
        ],
        dtype=float,
    ).T
    cells = np.vstack(
        [
            (
                np.array([[i, i + 1, i + n + 2], [i + n + 2, i + n + 1, i]])
                if (i + 1) % (n + 1) != 0
                else np.zeros((0, 3), dtype=int)
            )
            for i in np.arange(vertices_vd.shape[0] - n - 1, dtype=int)
        ],
        dtype=int,
    )    

    v_vu = jnp.array(np.random.uniform(size=vertices_vd.shape))
    v_g = v_vu.flatten()


    print("Creating Assembly Maps")
    # jax.profiler.start_trace("prof/assembly_map_prof")
    [sparse_assembly_map, index_assembly_map] = (
        benchmark_comparator(
            {
                "sparse": mesh_to_sparse_assembly_map,
                "index": mesh_to_index_assembly_map,
            },
            assert_equal=False,
            n_vertices=vertices_vd.shape[0],
            cells=cells,
        )
    )
    assembly_maps = {
        "sparse": sparse_assembly_map,
        "index": index_assembly_map,
    }
    print("V to EN")
    print("raveled")
    v_end = benchmark_comparator(
        {
            "sparse": lambda assembly_map,v_g: sparse_transform_global_to_element_node(assembly_map,v_g,E=cells.shape[0]),
            "index": lambda assembly_map,v_g: index_transform_global_to_element_node(assembly_map,v_g),
        },
        assembly_maps=assembly_maps,
        v_g=v_vu,
    )
    print("unraveled")
    v_end2 = benchmark_comparator(
        {
            "sparse": lambda assembly_map,v_g: sparse_transform_global_unraveled_to_element_node(assembly_map,v_g,E=cells.shape[0]),
            "index": lambda assembly_map,v_g: index_transform_global_unraveled_to_element_node(assembly_map,v_g),
        },
        assembly_maps=assembly_maps,
        v_g=v_g,
    )
    assert jnp.allclose(v_end, v_end2)
    print("EN to V")
    print("nosum")
    v_vu = benchmark_comparator(
        {
            "sparse": sparse_transform_element_node_to_global_nosum,
            "index": index_transform_element_node_to_global_nosum,
        },
        assembly_maps=assembly_maps,
        v_en=v_end,
    )
    print("unraveled nosum")
    v_g = benchmark_comparator(
        {
            "sparse": sparse_transform_element_node_to_global_unraveled_nosum,
            "index": index_transform_element_node_to_global_unraveled_nosum,
        },
        assembly_maps=assembly_maps,
        v_en=v_end,
    )
    print("sum")
    v_vu = benchmark_comparator(
        {
            "sparse": sparse_transform_element_node_to_global_sum,
            "index": index_transform_element_node_to_global_sum,
        },
        assembly_maps=assembly_maps,
        v_en=v_end,
    )
    print("unraveled sum")
    v_g = benchmark_comparator(
        {
            "sparse": sparse_transform_element_node_to_global_unraveled_sum,
            "index": index_transform_element_node_to_global_unraveled_sum,
        },
        assembly_maps=assembly_maps,
        v_en=v_end,
    )
    # jax.profiler.stop_trace()

    if derivatives:
        onehot_g = jnp.zeros_like(v_g).at[jnp.array([3,5,7,11])].set(1)
        onehot_en = jnp.zeros_like(v_end).at[jnp.array([3,5,7,11]),jnp.array([2,1,1,0]),0].set(1)
        print("JVPs")
        print("V to EN")
        J_sparse_end = benchmark_comparator(
            {
                "sparse": jax.jit(
                    lambda v_g,assembly_map,d_g: jax.jvp(
                        lambda v_g: sparse_transform_global_unraveled_to_element_node(
                            assembly_map=assembly_map, v_g=v_g, E=cells.shape[0]
                        ),
                        (v_g,),
                        (d_g,),
                    )[1]
                ),
                "index": jax.jit(
                    lambda v_g,assembly_map,d_g: jax.jvp(
                        lambda v_g: index_transform_global_unraveled_to_element_node(
                            assembly_map=assembly_map, v_g=v_g
                        ),
                        (v_g,),
                        (d_g,),
                    )[1]
                ),
            },
            assembly_maps=assembly_maps,
            v_g=v_g, 
            d_g=onehot_g       
        )

        print("EN to V")
        J_sparse_g = benchmark_comparator(
            {
                "sparse":jax.jit(
                    lambda v_en,assembly_map,d_en: jax.jvp(
                        lambda v_en: sparse_transform_element_node_to_global_unraveled_sum(
                            assembly_map=assembly_map, v_en=v_en
                        ),
                        (v_en,),
                        (d_en,),
                    )[1]
                ),
                "index":jax.jit(
                    lambda v_en,assembly_map,d_en: jax.jvp(
                        lambda v_en: index_transform_element_node_to_global_unraveled_sum(
                            assembly_map=assembly_map, v_en=v_en
                        ),
                        (v_en,),
                        (d_en,),
                    )[1]
                )
            },
            assembly_maps=assembly_maps,
            v_en=v_end,
            d_en=onehot_en
        )
        print("VJPs")
        print("V to EN")
        J_sparse_end = benchmark_comparator(
            {
                "sparse": jax.jit(
                    lambda v_g,assembly_map,d_en: jax.vjp(
                        lambda v_g: sparse_transform_global_unraveled_to_element_node(
                            assembly_map=assembly_map, v_g=v_g, E=cells.shape[0]
                        ),
                        v_g,
                    )[1](d_en)[0]
                ),
                "index": jax.jit(
                    lambda v_g,assembly_map,d_en: jax.vjp(
                        lambda v_g: index_transform_global_unraveled_to_element_node(
                            assembly_map=assembly_map, v_g=v_g
                        ),
                        v_g,
                    )[1](d_en)[0]
                ),
            },
            assembly_maps=assembly_maps,
            v_g=v_g,
            d_en=onehot_en    
        )
        print("EN to V")
        J_sparse_g = benchmark_comparator(
            {
                "sparse":jax.jit(
                    lambda v_en,assembly_map,d_g: jax.vjp(
                        lambda v_en: sparse_transform_element_node_to_global_unraveled_sum(
                            assembly_map=assembly_map, v_en=v_en
                        ),
                        v_en,
                    )[1](d_g)[0]
                ),
                "index":jax.jit(
                    lambda v_en,assembly_map,d_g: jax.vjp(
                        lambda v_en: index_transform_element_node_to_global_unraveled_sum(
                            assembly_map=assembly_map, v_en=v_en
                        ),
                        v_en,
                    )[1](d_g)[0]
                )
            },
            assembly_maps=assembly_maps,
            v_en=v_end,
            d_g=onehot_g
        )


