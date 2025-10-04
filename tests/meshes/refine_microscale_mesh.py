import meshio
import gmsh
import numpy as np
import os

# Change directories to the test/meshes dir
script_directory = os.path.dirname(__file__)
os.chdir(script_directory)

mesh = meshio.read(f"microscale_2D.vtk")

assert len(mesh.cells) == 1, "The current algorithm assumes only one type of cells in the VTK file."

gmsh.initialize()

cell_type = mesh.cells[0].type
cells = np.array(mesh.cells[0].data, dtype=np.uint64)
domain_ids = mesh.cell_data['DomainIDs'][0]
unique_domains = np.unique(domain_ids)

# Create a discrete entity to hold all nodes
# Note: adjustment for 1-based
node_tags = np.arange(1, mesh.points.shape[0] + 1)
node_entity_tag = unique_domains[-1] + 1
gmsh.model.add_discrete_entity(dim=2, tag=node_entity_tag)
gmsh.model.mesh.add_nodes(
    dim=2,
    tag=node_entity_tag,
    nodeTags=node_tags,
    coord=mesh.points.flatten()
)

match cell_type:
    case 'triangle3':
        gmsh_cell_type = gmsh.model.mesh.get_element_type("Triangle", 1)
    case 'triangle6':
        gmsh_cell_type = gmsh.model.mesh.get_element_type("Triangle", 2)
    case _:
        raise Exception(f'Case for gmsh cell type that matches vtk type {cell_type} is missing.')

print('gmsh_cell_type', gmsh_cell_type)

# Create a new discrete entity for each domain
for domain_id in unique_domains:
    gmsh.model.add_discrete_entity(dim=2, tag=domain_id)
    cell_indices_for_domain = np.where(domain_ids == domain_id)[0]
    # Note: adjustment for 1-based indices
    connectivity_for_domain = cells[cell_indices_for_domain, :].flatten() + 1
    # Note: element tags need to be unique across all domains
    start_tag = gmsh.model.mesh.get_max_element_tag() + 1
    element_tags = np.arange(start_tag, start_tag + len(cell_indices_for_domain))

    # Add the elements to the discrete surface we just created
    gmsh.model.mesh.add_elements_by_type(domain_id, gmsh_cell_type, element_tags, connectivity_for_domain)

gmsh.write(f"microscale_2D.msh")

def update_meshio_from_gmsh(mesh):
    # Extract the points and cells from the gmsh model and convert back to the VTK data structure
    node_tags, coords, _ = gmsh.model.mesh.get_nodes(dim=2)
    mesh.points = coords.reshape(len(node_tags), 3)

    cells = []
    domain_ids = []
    for domain_id in unique_domains:
        elem_types, elem_tags, node_tags = gmsh.model.mesh.get_elements(dim=2, tag=domain_id)
        node_tags = np.array(node_tags[0])
        # Note: adjustment back to 0-based
        cells.append(node_tags - 1)
        print(domain_id, elem_types, len(elem_tags[0]), node_tags.shape)
        domain_ids.append([domain_id] * len(elem_tags[0]))

    cells = np.hstack(cells)
    domain_ids = np.hstack(domain_ids)
    print(cells.shape)

    mesh.cells[0].data = cells.reshape(len(domain_ids), mesh.cells[0].data.shape[1])
    new_cell_data = mesh.cell_data
    new_cell_data["DomainIDs"][0] = np.array(domain_ids, dtype=np.int64)
    mesh.cell_data = new_cell_data


# Refine the mesh a few times
gmsh.model.mesh.refine()
gmsh.model.mesh.set_order(2)
update_meshio_from_gmsh(mesh)
meshio.write("microscale_2D_r1.vtk", mesh)

gmsh.model.mesh.refine()
gmsh.model.mesh.set_order(2)
update_meshio_from_gmsh(mesh)
meshio.write("microscale_2D_r2.vtk", mesh)

gmsh.model.mesh.refine()
gmsh.model.mesh.set_order(2)
update_meshio_from_gmsh(mesh)
meshio.write("microscale_2D_r3.vtk", mesh)

gmsh.finalize()