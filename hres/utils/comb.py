import numpy as np
from hres.utils.interpolation import interp2x

def comb(s, x, t):
    """
    Interpolate gas properties at current chamber pressure and O/F ratio
    
    Parameters:
    -----------
    s : dict
        Simulation parameters
    x : dict
        Current simulation state
    t : float
        Current time
        
    Returns:
    --------
    dict
        Updated simulation state
    """
    # Only update gas properties if within burn time or if burn time is unlimited
    if t <= s['tburn'] or s['tburn'] == 0:
        # Interpolate specific heat ratio
        x['k'] = interp2x(s['prop_OF'], s['prop_Pc'], s['prop_k'], x['OF'], x['P_cmbr'])
        
        # Interpolate molecular mass
        x['M'] = interp2x(s['prop_OF'], s['prop_Pc'], s['prop_M'], x['OF'], x['P_cmbr'])
        
        # Interpolate adiabatic flame temperature
        x['T'] = interp2x(s['prop_OF'], s['prop_Pc'], s['prop_T'], x['OF'], x['P_cmbr'])
        
        # Calculate gas constant (J/kg-K)
        x['R'] = 8314.5 / x['M']
        
        # Calculate density
        x['rho'] = x['P_cmbr'] / (x['R'] * x['T'])
        
        # Calculate characteristic velocity
        x['cstar'] = s['cstar_eff'] * np.sqrt((x['R'] * x['T']) / 
                                             (x['k'] * (2/(x['k']+1))**((x['k']+1)/(x['k']-1))))
    
    return x
