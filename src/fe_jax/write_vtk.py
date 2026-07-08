import pyvista as pv
import jax.numpy as jnp
import numpy as np

def write2VTK_avg(args,vtk_mesh,u_full,internal_state,fiber_tri_id,matrix_tri_id,fiber_quad_id,matrix_quad_id):
    '''This is the version that uses only all quadrature for saving and average value of all quadratures'''
    # Displacement
    vtk_mesh['displacement'] = u_full
    for idx, id in enumerate([matrix_tri_id,matrix_quad_id,fiber_tri_id,fiber_quad_id]):
        # internal_state[idx] has shape (num_elements, num_quad_points, num_state_vars)
        state_q = np.array(internal_state[idx])
        
        # Strains are at 0:3, Stresses are at 3:6, Damage is at 6
        strain_q = state_q[:, :, 0:3]
        stress_eff_q = state_q[:, :, 3:6]
        damage_q = state_q[:, :, 6]
        
        # Apply damage to stress at each quadrature point (only for matrix elements)
        if idx < 2:
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