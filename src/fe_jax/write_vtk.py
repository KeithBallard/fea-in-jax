import pyvista as pv
import jax.numpy as jnp

def write2VTK(args,vtk_mesh,u_full,ISV_eqi,fiber_tri_id,matrix_tri_id,enrich_tri_id,enrich_quad_id):
    '''This is the version that uses only one quadature for saving and does not save the average value of all quadatures'''
    # TODO the index in this function needs to be updated !!!!
    # Displacement
    vtk_mesh['displacement'] = u_full
    # Damage
    vtk_mesh.cell_data['damage'][matrix_tri_id]  = ISV_eqi[0][:,0,1]  # Matrix
    vtk_mesh.cell_data['damage'][enrich_tri_id]  = ISV_eqi[1][:,0,1]  # Matrix
    vtk_mesh.cell_data['damage'][enrich_quad_id] = ISV_eqi[1][:,1,1]  # Matrix
    # vtk_mesh.cell_data['damage'][enrich_quad_id] = jnp.mean(ISV_eqi[1][:,1:,1],axis=1)  # Matrix
    # Strain
    stress_key = ['e11','e22','e12','s11','s22','s12']
    for idx,key in enumerate(stress_key):
        vtk_mesh.cell_data[key][matrix_tri_id]  = ISV_eqi[0][:, 0, 5+idx]
        vtk_mesh.cell_data[key][enrich_tri_id]  = ISV_eqi[1][:, 0, 5+idx]
        # vtk_mesh.cell_data[key][enrich_quad_id] = ISV_eqi[1][:, 1, 5+idx]
        vtk_mesh.cell_data[key][enrich_quad_id] = jnp.mean(ISV_eqi[1][:, 1:, 5+idx],axis=1)
        vtk_mesh.cell_data[key][fiber_tri_id]   = ISV_eqi[2][:, 0, 5+idx]

    return vtk_mesh


def write2VTK_avg(args,vtk_mesh,u_full,element_batches,fiber_tri_id,matrix_tri_id,fiber_quad_id,matrix_quad_id):
    '''This is the version that uses only all quadature for saving and average value of all quadatures'''
    # Displacement
    vtk_mesh['displacement'] = u_full
    for idx, id in enumerate([matrix_tri_id,matrix_quad_id,fiber_tri_id,fiber_quad_id]):
        # Damage
        cell_damage = jnp.mean(element_batches[idx].internal_state[:,:,6], axis=1)
        vtk_mesh.cell_data['damage'][id] = cell_damage
        # Strain
        feature_key = ['e11','e22','e12','s11','s22','s12']
        for idx_i,key in enumerate(feature_key):
            if 's' in key:
                vtk_mesh.cell_data[key][id]  = jnp.mean(element_batches[idx].internal_state[:,:,idx_i], axis=1) * (1 - cell_damage)
            else:
                vtk_mesh.cell_data[key][id]  = jnp.mean(element_batches[idx].internal_state[:,:,idx_i], axis=1)

    return vtk_mesh


