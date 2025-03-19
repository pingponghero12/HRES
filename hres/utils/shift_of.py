import numpy as np

def shift_of(s, x):
    """
    Model regression of fuel grain using exponential regression law
    
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
    
    # Calculate port area and oxidizer mass flux
    A = 0.25 * np.pi * x['grn_ID']**2
    G = x['mdot_o'] / A
    
    # Calculate regression rate using exponential regression law
    # rdot = a * G^n * L^m (converted from mm/s to m/s)
    x['rdot'] = 0.001 * s['prop_Reg'][0] * G**s['prop_Reg'][1] * s['grn_L']**s['prop_Reg'][2]
    
    # Calculate fuel mass flow rate
    x['mdot_f'] = s['prop_Rho'] * x['rdot'] * np.pi * x['grn_ID'] * s['grn_L']
    
    # Calculate O/F ratio
    if x['mdot_f'] > 0:
        x['OF'] = x['mdot_o'] / x['mdot_f']
    else:
        x['OF'] = 0
    
    # Update grain geometry
    x['grn_ID_old'] = x['grn_ID']
    x['grn_ID'] = x['grn_ID'] + 2 * x['rdot'] * dt
    
    # Update fuel mass
    x['m_f'] = x['m_f'] - x['mdot_f'] * dt
    
    return x
