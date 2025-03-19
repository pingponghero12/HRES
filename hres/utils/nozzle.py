import numpy as np
from scipy import optimize

def nozzle(s, x):
    """
    Calculate thrust for the current timestep
    
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
    # Only calculate thrust if chamber pressure exceeds ambient pressure
    if x['P_cmbr'] > s['Pa']:
        # Function to solve for Mach number at nozzle exit
        def A_Ratio(M):
            return (((x['k']+1)/2)**(-((x['k']+1)/(2*(x['k']-1)))) * 
                   (1+(x['k']-1)/2*M**2)**((x['k']+1)/(2*(x['k']-1))) / M - 
                   s['noz_ER'])
        
        # Solve for Mach number (initial guess of 3 is common for supersonic nozzles)
        M = optimize.fsolve(A_Ratio, 3.0)[0]
        
        # Calculate exit pressure
        Pe = x['P_cmbr'] * (1 + 0.5 * (x['k']-1) * M**2)**(-x['k']/(x['k']-1))
        
        # Calculate thrust coefficient
        Cf = np.sqrt(((2*x['k']**2)/(x['k']-1)) * 
                     (2/(x['k']+1))**((x['k']+1)/(x['k']-1)) * 
                     (1-(Pe/x['P_cmbr'])**((x['k']-1)/x['k']))) + \
             ((Pe-s['Pa'])*(0.25*np.pi*s['noz_thrt']**2*s['noz_ER'])) / \
             (x['P_cmbr']*0.25*np.pi*s['noz_thrt']**2)
        
        # Calculate thrust
        x['F_thr'] = s['noz_eff'] * Cf * 0.25 * np.pi * s['noz_thrt']**2 * x['P_cmbr'] * s['noz_Cd']
        
        # Ensure thrust is non-negative
        if x['F_thr'] < 0:
            x['F_thr'] = 0
    else:
        x['F_thr'] = 0
    
    return x
