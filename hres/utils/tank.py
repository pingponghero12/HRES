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
    
    # Store previous state values
    m_o_old = x['m_o']
    mLiq_old = x['mLiq_new']  # Save the current liquid mass as the old value
    T_tnk_old = x['T_tnk']
    
    # Find oxidizer thermophysical properties
    x['ox_props'] = nox(x['T_tnk'])
    x['P_tnk'] = x['ox_props']['Pv']
    
    # Find oxidizer mass flow rate
    dP = x['P_tnk'] - x['P_cmbr']
    
    # Ensure pressure difference is positive
    if dP < 0:
        dP = 0
    
    # Calculate mass flow rates based on burn time and vent state
    if s['tburn'] == 0 or t <= s['tburn']:
        if s['vnt_S'] == 0:
            x['mdot_v'] = 0
            if x['mLiq_new'] == 0:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_v'] * dP)
            else:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_l'] * dP)
            
        elif s['vnt_S'] == 1:
            x['mdot_v'] = s['vnt_CdA'] * np.sqrt(2 * x['ox_props']['rho_v'] * dP)
            if x['mLiq_new'] == 0:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_v'] * dP)
            else:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_l'] * dP)
            
        elif s['vnt_S'] == 2:
            x['mdot_v'] = s['vnt_CdA'] * np.sqrt(2 * x['ox_props']['rho_v'] * dP)
            if x['mLiq_new'] == 0:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_v'] * dP) + x['mdot_v']
            else:
                x['mdot_o'] = s['inj_CdA'] * s['inj_N'] * np.sqrt(2 * x['ox_props']['rho_l'] * dP) + x['mdot_v']
            
        else:
            raise ValueError('Error: Vent State Undefined')
            
    elif s['tburn'] > 0 and t > s['tburn']:
        x['mdot_o'] = 0
        x['mdot_v'] = 0
    
    # Update total oxidizer mass
    mass_discharged = x['mdot_o'] * dt
    x['m_o'] = x['m_o'] - mass_discharged
    
    # Enable debug output for testing
    debug = False
    
    # Handle two-phase (liquid-vapor) state
    if mLiq_old > 0 and x['mdot_o'] > 0:
        # CRITICAL FIX: Calculate liquid loss directly
        liquid_discharged = mass_discharged
        
        # Calculate new liquid mass after discharge (before evaporation)
        mLiq_after_discharge = mLiq_old - liquid_discharged
        
        if debug:
            print(f"t={t:.3f}s: Initial mLiq={mLiq_old:.6f}kg, Discharge={liquid_discharged:.6f}kg")
        
        # Calculate how much liquid should remain based on volume balance
        x['ox_props'] = nox(x['T_tnk'])  # Ensure properties are up to date
        
        # Volume balance equation: V_tank = V_liquid + V_vapor
        # V_liquid = m_liquid / rho_liquid
        # V_vapor = m_vapor / rho_vapor = (m_o - m_liquid) / rho_vapor
        # Solve for m_liquid:
        mLiq_new_ideal = (s['tnk_V'] - (x['m_o']/x['ox_props']['rho_v'])) / \
                         ((1/x['ox_props']['rho_l']) - (1/x['ox_props']['rho_v']))
        
        # Ensure non-negative liquid mass
        mLiq_new_ideal = max(0, mLiq_new_ideal)
        
        if debug:
            print(f"  After discharge: {mLiq_after_discharge:.6f}kg, Ideal: {mLiq_new_ideal:.6f}kg")
        
        # If we need more evaporation to maintain pressure equilibrium
        if mLiq_new_ideal < mLiq_after_discharge and mLiq_new_ideal > 0:
            # Calculate additional liquid that must evaporate to maintain pressure
            m_evaporated = mLiq_after_discharge - mLiq_new_ideal
            
            # Calculate temperature change due to evaporative cooling
            # Energy balance: m_evaporated * Hv = m_remaining * Cp * dT
            dT = -m_evaporated * x['ox_props']['Hv'] / (mLiq_new_ideal * x['ox_props']['Cp'])
            
            if debug:
                print(f"  Additional evaporation: {m_evaporated:.6f}kg, dT={dT:.3f}K")
            
            # Update temperature
            x['T_tnk'] = x['T_tnk'] + dT
            
            # Update properties at new temperature
            x['ox_props'] = nox(x['T_tnk'])
            x['P_tnk'] = x['ox_props']['Pv']
            
            # Final liquid mass after evaporation
            x['mLiq_new'] = mLiq_new_ideal
            
            # Calculate and store pressure change for reference
            x['dP'] = x['P_tnk'] - x['ox_props']['Pv']
            
        else:
            # Either no additional evaporation needed or all liquid evaporated
            x['mLiq_new'] = mLiq_after_discharge if mLiq_after_discharge > 0 else 0
            
            if x['mLiq_new'] == 0:
                if debug:
                    print("  All liquid evaporated - transitioning to gas phase")
        
        # Update mLiq_old for next step
        x['mLiq_old'] = mLiq_old
        
    # Gas-only phase
    elif mLiq_old <= 0 and x['mdot_o'] > 0:
        # Ensure liquid is exactly 0
        x['mLiq_new'] = 0
        x['mLiq_old'] = 0
        
        # Find Z factor using iterative method
        Z_old = x['ox_props']['Z']
        Zguess = Z_old
        epsilon = 1
        
        Ti = x['T_tnk']
        Pi = x['P_tnk']
        
        # Iteratively solve for Z factor
        iter_count = 0
        while epsilon >= 0.000001 and iter_count < 50:  # Add iteration limit
            iter_count += 1
            
            try:
                T_ratio = ((Zguess * x['m_o']) / (Z_old * m_o_old))**(0.3)
                x['T_tnk'] = T_ratio * Ti
                P_ratio = T_ratio**(1.3/0.3)
                x['P_tnk'] = P_ratio * Pi
                
                x['ox_props'] = nox(x['T_tnk'])
                Z = x['ox_props']['Z']
                
                epsilon = abs(Zguess - Z)
                Zguess = (Zguess + Z) / 2
                
            except Exception as e:
                print(f"Error in Z-factor calculation: {e}")
                break
    
    if debug and abs(x['T_tnk'] - T_tnk_old) > 0.01:
        print(f"  Temperature changed: {T_tnk_old:.2f}K → {x['T_tnk']:.2f}K")
        print(f"  Pressure: {x['P_tnk']/1e6:.3f}MPa")
    
    return x
