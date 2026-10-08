import pyvista as pv
import jax.numpy as jnp
import numpy as np

def write2VTK_ISV(args,vtk_mesh,u,ISV_be,cell_ids_b):
    '''
    ISV_be[idx] has shape (num_elements, 7) and is already averaged with damage applied.
    cell_ids_b[idx] holds the mesh cell index of each element of batch idx (same order as ISV_be).
    '''
    # Displacement
    vtk_mesh['displacement'][:,:2] = u.reshape(-1, 2)
    for idx, id in enumerate(cell_ids_b):
        avg_state = np.array(ISV_be[idx])
        
        avg_strain = avg_state[:, 0:3]
        avg_stress_dmg = avg_state[:, 3:6]
        avg_damage = avg_state[:, 6]
        
        vtk_mesh.cell_data['damage'][id] = avg_damage
        feature_key = ['e11','e22','e12','s11','s22','s12']
        for idx_i, key in enumerate(feature_key):
            if 's' in key:
                vtk_mesh.cell_data[key][id] = avg_stress_dmg[:, idx_i - 3]
            else:
                vtk_mesh.cell_data[key][id] = avg_strain[:, idx_i]

    return vtk_mesh


def write2VTK_avg(args,vtk_mesh,u,element_batches,cell_ids_b):
    '''
    This is the version that uses only all quadrature for saving and average value of all quadratures.
    cell_ids_b[idx] holds the mesh cell index of each element of element_batches[idx].
    '''
    # Displacement
    vtk_mesh['displacement'][:,:2] = u.reshape(-1, 2)
    # vtk_mesh['displacement'] = u_full
    for idx, id in enumerate(cell_ids_b):
        # internal_state[idx] has shape (num_elements, num_quad_points, num_state_vars)
        state_q = np.array(element_batches[idx].internal_state)
        
        # Strains are at 0:3, Stresses are at 3:6, Damage is at 6
        strain_q = state_q[:, :, 0:3]
        stress_eff_q = state_q[:, :, 3:6]
        damage_q = state_q[:, :, 6]
        
        # TODO double check if this is still needed. might be double dipping in fea.py
        # Apply damage to stress at each quadrature point (only for matrix elements, which are the
        # batches with the 11 damage-model state variables; fibers have 7, see compute_ISV_be)
        if state_q.shape[-1] >= 8:
            stress_dmg_q = stress_eff_q * (1 - damage_q[:, :, np.newaxis])
        else:
            stress_dmg_q = stress_eff_q
        
        # Now average over quadrature points (axis 1)
        avg_strain = np.mean(strain_q, axis=1)
        avg_stress_dmg = np.mean(stress_dmg_q, axis=1)
        avg_damage = np.mean(damage_q, axis=1)
        
        vtk_mesh.cell_data['damage'][id] = avg_damage
        feature_key = ['e11','e22','e12','s11','s22','s12']
        for idx_i, key in enumerate(feature_key):
            if 's' in key:
                vtk_mesh.cell_data[key][id] = avg_stress_dmg[:, idx_i - 3]
            else:
                vtk_mesh.cell_data[key][id] = avg_strain[:, idx_i]

    return vtk_mesh