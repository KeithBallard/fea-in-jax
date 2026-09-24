import shutil
from pathlib import Path
import pyvista as pv

import adios2
import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np


# Verify JAX is running on the GPU
assert jax.devices()[0].platform == "gpu", \
    "JAX is not utilizing the GPU!"


def remove_existing_bp(path):
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_unstructured_polygon_adios2_vtx(
    out_bp="IGFEM_mesh_vtx.bp",
):


    """
    Write a mixed triangle/quad-shaped 2D mesh using VTK_POLYGON cells.

    Important ADIOS2VTXReader limitations
    -------------------------------------
    1. The reader supports only one scalar VTK cell type.
    2. Therefore, all cells are stored as VTK_POLYGON = 7.
    3. Connectivity must use the legacy VTK layout:

           [number_of_points, point_0, point_1, ...]

       A separate offsets array must not be used.

    Array-valued mesh and field variables are passed to ADIOS2 directly
    from GPU memory using JAX -> DLPack -> CuPy.
    """

    remove_existing_bp(out_bp)

    mesh   = pv.read('IGFEM_1fib.vtk')
    # ------------------------------------------------------------------
    # Mesh coordinates on the GPU
    # ------------------------------------------------------------------
    vertices_JAX = jnp.array(mesh.points, dtype=np.float32)

    # ------------------------------------------------------------------
    # Legacy VTK connectivity on the GPU
    # ------------------------------------------------------------------
    cells_JAX  = jnp.array(mesh.cells, dtype=np.uint32)
    # Packed legacy VTK connectivity.
    connectivity_JAX = jnp.array([mesh.cells])

    # Point data: one value per mesh point
    Point_data = ['displacement']
    point_data_JAX = {key: jnp.array(mesh[key]) for key in Point_data}

    # Cell data: one value per polygon cell
    feature_key = ['e11','e22','e12','s11','s22','s12','damage']
    cell_data_JAX = {key: jnp.array(mesh[key]) for key in feature_key}

    # ------------------------------------------------------------------
    # Scalar metadata remains on the CPU
    # ------------------------------------------------------------------
    VTK_POLYGON = np.array(7, dtype=np.uint32)
    num_vertices = np.array(vertices_JAX.shape[0], dtype=np.uint32)
    num_elements = np.array(cells_JAX.shape[0], dtype=np.uint32)

    # ------------------------------------------------------------------
    # VTK XML descriptor
    # ------------------------------------------------------------------
    point_data_xml = ""
    if Point_data:
        point_data_xml = f'      <PointData Vectors="{Point_data[0]}">\n'
        for key in Point_data:
            point_data_xml += f'        <DataArray Name="{key}" />\n'
        point_data_xml += '      </PointData>'

    cell_data_xml = ""
    if feature_key:
        cell_data_xml = f'      <CellData Scalars="{feature_key[0]}">\n'
        for key in feature_key:
            cell_data_xml += f'        <DataArray Name="{key}" />\n'
        cell_data_xml += '      </CellData>'

    vtk_xml = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid"
         version="0.1"
         byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="NumOfVertices"
           NumberOfCells="NumOfElements">
      <Points>
        <DataArray Name="vertices" />
      </Points>
      <Cells>
        <DataArray Name="connectivity" />
        <DataArray Name="types" />
      </Cells>
{point_data_xml}
{cell_data_xml}
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    # ------------------------------------------------------------------
    # Initialize ADIOS2
    # ------------------------------------------------------------------
    adios = adios2.Adios()
    io = adios.declare_io("VTXWriter")

    # Scalar metadata variables
    var_num_vertices = io.define_variable(
        "NumOfVertices",
        np.zeros([], dtype=np.uint32),
    )

    var_num_elements = io.define_variable(
        "NumOfElements",
        np.zeros([], dtype=np.uint32),
    )

    # ADIOS2VTXReader requires a single scalar type.
    var_types = io.define_variable(
        "types",
        np.zeros([], dtype=np.uint32),
    )
    # ADIOS2's Python binding reverses multidimensional count ordering.
    # Therefore, pass the reversed shape to obtain the intended C++/VTK shape.

    var_vertices = io.define_variable(
        "vertices",
        np.zeros(vertices_JAX.shape, dtype=vertices_JAX.dtype),
        [],
        [],
        list(reversed(vertices_JAX.shape)),       # [3, # of nodes] -> stored as (# of nodes, 3)
    )

    var_connectivity = io.define_variable(
        "connectivity",
        np.zeros(connectivity_JAX.shape, dtype=connectivity_JAX.dtype),
        [],
        [],
        list(reversed(connectivity_JAX.shape)),   # [len(cells), 1] -> stored as (1, len(cells))
    )

    point_vars = {}
    for key, jax_arr in point_data_JAX.items():
        shape = list(reversed(jax_arr.shape)) if jax_arr.ndim > 1 else list(jax_arr.shape)
        point_vars[key] = io.define_variable(
            key,
            np.zeros(jax_arr.shape, dtype=jax_arr.dtype),
            [],
            [],
            shape,
        )

    cell_vars = {}
    for key, jax_arr in cell_data_JAX.items():
        shape = list(reversed(jax_arr.shape)) if jax_arr.ndim > 1 else list(jax_arr.shape)
        cell_vars[key] = io.define_variable(
            key,
            np.zeros(jax_arr.shape, dtype=jax_arr.dtype),
            [],
            [],
            shape,
        )

    # ------------------------------------------------------------------
    # Use BP5, which supports GPU-resident input buffers.
    io.set_engine("BP5")
    # Store the VTK descriptor inside the BP dataset.
    io.define_attribute("vtk.xml", vtk_xml)

    # ------------------------------------------------------------------
    # Zero-copy JAX -> CuPy conversion through DLPack
    # ------------------------------------------------------------------
    cupy_vertices = cp.from_dlpack(vertices_JAX)
    cupy_connectivity = cp.from_dlpack(connectivity_JAX)
    cupy_point_data = {key: cp.from_dlpack(arr) for key, arr in point_data_JAX.items()}
    cupy_cell_data = {key: cp.from_dlpack(arr) for key, arr in cell_data_JAX.items()}

    # ------------------------------------------------------------------
    # Write the BP dataset
    # ------------------------------------------------------------------
    writer = io.open(out_bp, adios2.Mode.Write)



    writer.begin_step()

    # CPU scalar metadata
    writer.put(var_num_vertices, num_vertices)
    writer.put(var_num_elements, num_elements)
    writer.put(var_types, VTK_POLYGON)

    # GPU-resident arrays
    writer.put(var_vertices, cupy_vertices)
    writer.put(var_connectivity, cupy_connectivity)
    for key in point_vars:
        writer.put(point_vars[key], cupy_point_data[key])
        
    for key in cell_vars:
        writer.put(cell_vars[key], cupy_cell_data[key])



    writer.end_step()



    writer.close()

    print(f"Wrote {out_bp}")
    print("Cell representation: VTK_POLYGON")
    print("Cells: 4 triangle-shaped polygons + 2 quad-shaped polygons")


if __name__ == "__main__":
    write_unstructured_polygon_adios2_vtx(
        "polygon_vtx.bp"
    )