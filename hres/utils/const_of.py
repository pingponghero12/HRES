import numpy as np

def const_of(s, x):
    """
    Model regression of fuel grain at a given value of OF
    
    Parameters:
    -----------
    s : dict
        Simulation configuration parameters
    x : dict
        Current simulation state
        
    Returns:
    --------
    dict
        Updated simulation state
    """
    dt = s['dt']
    
    # Calculate fuel mass flow rate based on constant OF ratio
    x['mdot_f'] = x['mdot_o'] / s['const_OF']
    
    # Calculate regression rate
    x['rdot'] = x['mdot_f'] / (s['prop_Rho'] * np.pi * x['grn_ID'] * s['grn_L'])
    
    # Save previous grain inner diameter
    x['grn_ID_old'] = x['grn_ID']
    
    # Update grain inner diameter
    x['grn_ID'] = x['grn_ID'] + 2 * x['rdot'] * dt
    
    # Update fuel mass
    x['m_f'] = x['m_f'] - x['mdot_f'] * dt
    
    return x
