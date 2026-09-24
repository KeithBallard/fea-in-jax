import meshio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from pathlib import Path
from matplotlib.colors import Normalize

def plot_mesh_to_png(mesh, feature,data_type, output_file="mesh_colored.png", dpi=300, cmap='coolwarm'):
    """
    Plots a 2D triangle or quad mesh colored by nodal field values (averaged per cell).
    
    Parameters:
        mesh_file (str or Path): Path to the mesh file (e.g., .vtk, .xdmf, .msh).
        feature (str): Name of the field to visualize (e.g., 'damage').
        output_file (str or Path): Output PNG file name.
        dpi (int): Resolution of the saved image.
        cmap (str): Matplotlib colormap name.
    """

    # Initialize mesh
    points    = mesh.points
    triangles = mesh.cells_dict.get("triangle")

    if data_type == 'points':
        nodal_values = mesh.point_data[feature]
        cell_values  = np.mean(nodal_values[:,0][triangles],axis=1)    
    if data_type == 'cells':
        cell_values  = mesh.cell_data[feature][1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    tpc = ax.tripcolor(
        points[:, 0], points[:, 1], triangles,
        facecolors=cell_values, edgecolors='k',
        linewidth=0.3, cmap=cmap
    )
    ax.set_aspect('equal')
    ax.axis('off')
    cbar = plt.colorbar(tpc, ax=ax, shrink=0.75)
    cbar.set_label(feature, fontsize=16) 
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi,bbox_inches='tight')
    plt.close()

def plot_IGFEM_mesh_to_png(mesh,i,feature,data_type, output_file="mesh_colored.png", dpi=300, cmap='coolwarm'):
    """
    Plots a 2D triangle or quad mesh colored by nodal field values (averaged per cell).
    
    Parameters:
        mesh_file (str or Path): Path to the mesh file (e.g., .vtk, .xdmf, .msh).
        feature (str): Name of the field to visualize (e.g., 'damage').
        output_file (str or Path): Output PNG file name.
        dpi (int): Resolution of the saved image.
        cmap (str): Matplotlib colormap name.
    """

    # Initialize mesh
    points = mesh.points[:,:2]

    # Go through the pyvista loop, which is absolute pain to do.
    polygons = []
    color = []
    for id in range(len(mesh.celltypes)):
        nodes = mesh.get_cell(id).point_ids
        polygons.append(points[nodes])
        if data_type == 'points':
            nodal_values = mesh.point_data[feature]
            cell_values  = np.mean(nodal_values[:,0][nodes])
            vmin, vmax = (-1.1e-6, 8.5e-5)

        if data_type == 'cells':
            cell_values  = mesh.cell_data[feature][id]
            vmin, vmax = (-0.2, 1.0)

        color.append(cell_values)

    # Plot
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    norm = Normalize(vmin=vmin, vmax=vmax)
    poly = PolyCollection(polygons, array=np.array(color), edgecolors='k', cmap=cmap,linewidth=0.3,norm=norm)
    ax.add_collection(poly)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')
    cbar = plt.colorbar(poly, ax=ax, shrink=0.75)
    cbar.set_label(feature, fontsize=16) 
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi,bbox_inches='tight')