import numpy as np
from scipy import optimize
from hres.utils.nox import nox

def tank(s, o, x, t):
    """
    Model oxidizer tank emptying in equilibrium
    
    Parameters:
    -----------
    s : dict
        Simulation parameters
    o : dict
        Output data history
    x : dict
        Current simulation state
    t : float
        Current time
        
    Returns:
    --------
    dict
        Updated simulation state
    """
    dt = s['dt']
    
    # Find oxidizer thermophysical properties
    x['ox_props'] = nox(x['T_tnk'])
    x['P_tnk'] = x['ox_props']['Pv']
    
    # Find oxidizer mass flow rate
    dP = x['P_tnk'] - x['P_cmbr']
    
    # Calculate Mach numbers
    Mcc = np.sqrt(x['ox_props']['Z'] * 1.31 * 188.91 * x['T_tnk'] * 
                 (x['P_cmbr']/x['P_tnk'])**(0.31/1.31))
    Matm = np.sqrt(x['ox_props']['Z'] * 1.31 * 188.91 * x['T_tnk'] * 
                  (s['Pa']/x['P_tnk'])**(0.31/1.31))
    
    # Limit Mach numbers to 1
    if Mcc >= 1:
        Mcc = 1
    
    if Matm >= 1:
        Matm = 1
    
    # Ensure pressure difference is positive
    if dP < 0:
        dP = 0
    
    # Calculate mass flow rates based on burn time and vent state
    if s['tburn'] == 0 or t <= s['tburn']:
        if s['vnt_S'] == 0:
            x['mdot_v'] = 0
            if x['mLiq_new'] == 0:
                x['mdot_o'] = (s['inj_CdA'] * s['inj_N'] * x['P_tnk'] / np.sqrt(x['T_tnk'])) * \
                             np.sqrt(1.31/(x['ox_props']['Z'] * 188.91)) * Mcc * \
                             (1 + (0.31)/2 * Mcc**2)**(-2.31/0.62)
            else:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_l'] * dP)
            mD = (x['mdot_o'] + x['mdot_v']) * dt
            
        elif s['vnt_S'] == 1:
            x['mdot_v'] = (s['vnt_CdA'] * x['P_tnk'] / np.sqrt(x['T_tnk'])) * \
                         np.sqrt(1.31/(x['ox_props']['Z'] * 188.91)) * Matm * \
                         (1 + (0.31)/2 * Matm**2)**(-2.31/0.62)
            if x['mLiq_new'] == 0:
                x['mdot_o'] = (s['inj_CdA'] * s['inj_N'] * x['P_tnk'] / np.sqrt(x['T_tnk'])) * \
                             np.sqrt(1.31/(x['ox_props']['Z'] * 188.91)) * Mcc * \
                             (1 + (0.31)/2 * Mcc**2)**(-2.31/0.62)
            else:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_l'] * dP)
            mD = (x['mdot_o'] + x['mdot_v']) * dt
            
        elif s['vnt_S'] == 2:
            x['mdot_v'] = (s['vnt_CdA'] * x['P_tnk'] / np.sqrt(x['T_tnk'])) * \
                         np.sqrt(1.31/(x['ox_props']['Z'] * 188.91)) * Matm * \
                         (1 + (0.31)/2 * Matm**2)**(-2.31/0.62)
            if x['mLiq_new'] == 0:
                x['mdot_o'] = (s['inj_CdA'] * s['inj_N'] * x['P_tnk'] / np.sqrt(x['T_tnk'])) * \
                             np.sqrt(1.31/(x['ox_props']['Z'] * 188.91)) * Mcc * \
                             (1 + (0.31)/2 * Mcc**2)**(-2.31/0.62)
            else:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_l'] * dP) + x['mdot_v']
            mD = x['mdot_o'] * dt
            
        else:
            raise ValueError('Error: Vent State Undefined')
            
    elif s['tburn'] > 0 and t > s['tburn']:
        x['mdot_o'] = 0
        mD = 0
    
    # Find mass discharged during time step
    m_o_old = x['m_o']
    x['m_o'] = x['m_o'] - x['mdot_o'] * dt
    
    # Handle different tank conditions based on liquid state
    if x['mLiq_new'] < x['mLiq_old'] and x['mLiq_new'] > 0 and x['mdot_o'] > 0:
        # Find mass of liquid nitrous evaporated during time step
        x['mLiq_old'] = x['mLiq_new'] - mD
        x['ox_props'] = nox(x['T_tnk'])
        x['mLiq_new'] = (s['tnk_V'] - (x['m_o']/x['ox_props']['rho_v'])) / \
                       ((1/x['ox_props']['rho_l']) - (1/x['ox_props']['rho_v']))
        mv = x['mLiq_old'] - x['mLiq_new']
        
        # Find heat removed from liquid
        dT = -mv * x['ox_props']['Hv'] / (x['mLiq_new'] * x['ox_props']['Cp'])
        x['T_tnk'] = x['T_tnk'] + dT
        op = nox(x['T_tnk'])
        x['dP'] = op['Pv'] - x['P_tnk']
        
    elif x['mLiq_new'] >= x['mLiq_old'] and x['mLiq_new'] > 0 and x['mdot_o'] > 0:
        # Calculate average pressure change
        valid_indices = [i for i, val in enumerate(o['dP']) if val < 0 and i < len(o['t']) and o['t'][i] > 0]
        if valid_indices:
            dP_avg = np.mean([o['dP'][i] for i in valid_indices])
        else:
            dP_avg = 0
        
        P_new = x['P_tnk'] + dP_avg
        
        # Define vapor pressure function for root finding
        def vapor_pressure(T):
            return 7251000 * np.exp((1/(T/309.57)) * (-6.71893*(1-T/309.57) + 
                  1.35966*(1-(T/309.57))**(3/2) + -1.3779*(1-(T/309.57))**(5/2) + 
                  -4.051*(1-(T/309.57))**5)) - P_new
        
        # Find temperature that gives the new pressure
        x['T_tnk'] = optimize.fsolve(vapor_pressure, x['T_tnk'])[0]
        x['dP'] = x['ox_props']['Pv'] - x['P_tnk']
        
        x['ox_props'] = nox(x['T_tnk'])
        x['mLiq_new'] = (s['tnk_V'] - (x['m_o']/x['ox_props']['rho_v'])) / \
                       ((1/x['ox_props']['rho_l']) - (1/x['ox_props']['rho_v']))
        x['mLiq_old'] = 0
        
    elif x['mLiq_new'] <= 0 and x['mdot_o'] > 0:
        # Make sure mLiq_new is exactly 0 if it's negative
        if x['mLiq_new'] != 0:
            x['mLiq_new'] = 0
        
        # Find Z factor using iterative method
        Z_old = x['ox_props']['Z']
        Zguess = Z_old
        epsilon = 1
        
        Ti = x['T_tnk']
        Pi = x['P_tnk']
        
        # Iteratively solve for Z factor
        while epsilon >= 0.000001:
            T_ratio = ((Zguess * x['m_o']) / (Z_old * m_o_old))**(0.3)
            x['T_tnk'] = T_ratio * Ti
            P_ratio = T_ratio**(1.3/0.3)
            x['P_tnk'] = P_ratio * Pi
            
            x['ox_props'] = nox(x['T_tnk'])
            Z = x['ox_props']['Z']
            
            epsilon = abs(Zguess - Z)
            Zguess = (Zguess + Z) / 2
    
    return x
