import numpy as np

def sim_loop(s, x, o, t):
    """
    Run the complete hres simulation loop
    
    Parameters:
    -----------
    s : dict
        Simulation parameters
    x : dict
        Initial simulation state
    o : dict
        Output data structure (pre-allocated)
    t : float
        Initial time
        
    Returns:
    --------
    tuple
        (s, x, o, t) - Final simulation parameters, state, output, and time
    """
    from hres.core.sim_iteration import sim_iteration
    
    i = 0  # Python uses 0-based indexing
    dt = s['dt']
    
    while True:
        # Update time and iteration counter
        t = i * dt
        i += 1
        
        # Run one simulation iteration
        s, x, o, t = sim_iteration(s, x, o, t, i)
        
        # Check for simulation end conditions
        if x['grn_ID'] >= s['grn_OD']:
            o['sim_end_cond'] = 'Fuel Depleted'
            break
        elif x['m_o'] <= 0:
            o['sim_end_cond'] = 'Oxidizer Depleted'
            break
        elif t >= s['tmax']:
            o['sim_end_cond'] = 'Max Simulation Time Reached'
            break
        elif x['P_cmbr'] <= s['Pa']:
            o['sim_end_cond'] = 'Burn Complete'
            break
    
    # Trim output arrays to actual data length
    # In MATLAB: o.t = o.t(1:sum(o.t>0)+1)
    # In Python, we count non-zero time entries (adjust for 0-based indexing)
    valid_length = np.sum(np.array(o['t']) > 0) + 1
    
    # Trim all output arrays to valid length
    for key in ['t', 'm_o', 'P_tnk', 'P_cmbr', 'mdot_o', 'mdot_f', 'OF', 
                'grn_ID', 'mdot_n', 'rdot', 'm_f', 'F_thr', 'dP']:
        o[key] = o[key][:valid_length]
    
    # Trim mass properties arrays if they were calculated
    if s['mp_calc'] == 1:
        o['m_t'] = o['m_t'][:valid_length]
        o['cg'] = o['cg'][:valid_length]
    
    return s, x, o, t
