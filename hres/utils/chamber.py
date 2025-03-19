import numpy as np

def chamber(s, x):
    """
    Calculate chamber pressure for the current timestep
    
    Parameters:
    -----------
    s : dict
        Simulation parameters
    x : dict
        Current simulation state
        
    Returns:
    --------
    dict
        Updated simulation state
    """
    dt = s['dt']
    
    # Calculate chamber volume
    if s['cmbr_V'] == 0:
        V = 0.25 * np.pi * x['grn_ID']**2 * s['grn_L']
    else:
        V = s['cmbr_V'] - 0.25 * np.pi * (s['grn_OD']**2 - x['grn_ID']**2) * s['grn_L']
    
    # Calculate volume change rate
    dV = 0.25 * np.pi * (x['grn_ID']**2 - x['grn_ID_old']**2) * s['grn_L'] / dt
    
    # Calculate nozzle mass flow rate
    x['mdot_n'] = x['P_cmbr'] * s['noz_Cd'] * 0.25 * np.pi * s['noz_thrt']**2 / x['cstar']
    
    # Calculate gas mass change rate
    dm_g = x['mdot_f'] + x['mdot_o'] - x['mdot_n']
    
    if x['mdot_o'] == 0:
        x['dm_g'] = -x['mdot_n']
    
    # Update gas mass
    x['m_g'] = x['m_g'] + dm_g * dt
    
    # Calculate pressure change rate
    dP = x['P_cmbr'] * (dm_g / x['m_g'] - dV / V)
    
    # Update chamber pressure
    x['P_cmbr'] = x['P_cmbr'] + dP * dt
    
    # Handle atmospheric pressure boundary condition
    if x['P_cmbr'] <= s['Pa']:
        x['P_cmbr'] = s['Pa']
        x['mdot_n'] = 0
    
    return x
