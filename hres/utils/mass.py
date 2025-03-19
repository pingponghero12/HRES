import numpy as np

def mass(s, x):
    """
    Calculate the total mass and center of gravity of the rocket motor
    
    Parameters:
    -----------
    s : dict
        Simulation parameters
    x : dict
        Current simulation state
        
    Returns:
    --------
    list
        [total_mass, center_of_gravity] - Total mass in kg, CG position in m
    """
    # Calculate total mass
    x['m_t'] = s['mtr_m'] + x['m_o'] + x['m_f']
    
    # Ensure liquid mass is non-negative
    if x['mLiq_new'] < 0:
        x['mLiq_new'] = 0
    
    # Calculate tank cross-sectional area
    tA = 0.25 * np.pi * s['tnk_D']**2
    
    if x['mLiq_new'] > 0:
        # Case with liquid present
        # Calculate vapor mass
        m_v = x['m_o'] - x['mLiq_new']
        
        # Calculate volumes
        vl = x['mLiq_new'] / x['ox_props']['rho_l']
        vv = s['tnk_V'] - vl
        
        # Calculate heights of liquid and vapor columns
        hl = vl / tA
        hv = vv / tA
        
        # Calculate centers of mass for components
        CoMl = s['tnk_X'] - hl / 2
        CoMv = s['tnk_X'] - hl - hv / 2
        CoMf = s['cmbr_X'] - s['grn_L'] / 2
        
        # Calculate overall center of gravity
        x['cg'] = (x['mLiq_new'] * CoMl + m_v * CoMv + x['m_f'] * CoMf + 
                  s['mtr_m'] * s['mtr_cg']) / x['m_t']
        
    elif x['mLiq_new'] == 0:
        # Case with no liquid present
        m_v = x['m_o']
        vv = s['tnk_V']
        hv = vv / tA
        
        # Calculate centers of mass
        CoMv = s['tnk_X'] - hv / 2
        CoMf = s['cmbr_X'] - s['grn_L'] / 2
        
        # Calculate overall center of gravity
        x['cg'] = (m_v * CoMv + x['m_f'] * CoMf + s['mtr_m'] * s['mtr_cg']) / x['m_t']
    
    # Return mass properties as a list [total_mass, center_of_gravity]
    return [x['m_t'], x['cg']]
