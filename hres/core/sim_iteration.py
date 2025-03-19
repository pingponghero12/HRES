def sim_iteration(s, x, o, t, i):
    """
    Execute one iteration of the hres simulation
    
    Parameters:
    -----------
    s : dict
        Simulation parameters
    x : dict
        Current simulation state
    o : dict
        Output data history
    t : float
        Current time
    i : int
        Current iteration index
        
    Returns:
    --------
    tuple
        (s, x, o, t) - Updated simulation parameters, state, output, and time
    """
    dt = s['dt']
    
    # Update time
    t = t + dt
    
    # Tank model - Update oxidizer state and mass flow rate
    from hres.utils.tank import tank
    x = tank(s, o, x, t)
    
    # Regression model - Update fuel regression based on specified model
    # This is a function pointer in the original MATLAB code
    # In Python, we'll use a dictionary of functions to mimic this behavior
    regression_models = {
        'const_of': lambda s, x: s['regression_model_func'](s, x),
        'shift_of': lambda s, x: s['regression_model_func'](s, x)
    }
    
    # Call the appropriate regression model function
    x = s['regression_model_func'](s, x)
    
    # Combustion model - Update gas properties
    from hres.utils.comb import comb
    x = comb(s, x, t)
    
    # Chamber model - Calculate chamber pressure
    from hres.utils.chamber import chamber
    x = chamber(s, x)
    
    # Nozzle model - Calculate thrust
    from hres.utils.nozzle import nozzle
    x = nozzle(s, x)
    
    # Optionally calculate mass properties
    if s['mp_calc'] == 1:
        from hres.utils.mass import mass
        mp = mass(s, x)
        o['m_t'][i] = mp[0]
        o['cg'][i] = mp[1]
    
    # Record current state in output history
    o['t'][i] = t
    o['m_o'][i] = x['m_o']
    o['P_tnk'][i] = x['P_tnk']
    o['P_cmbr'][i] = x['P_cmbr']
    o['mdot_o'][i] = x['mdot_o']
    o['mdot_f'][i] = x['mdot_f']
    o['OF'][i] = x['OF']
    o['grn_ID'][i] = x['grn_ID']
    o['mdot_n'][i] = x['mdot_n']
    o['rdot'][i] = x['rdot']
    o['m_f'][i] = x['m_f']
    o['dP'][i] = x.get('dP', 0)
    o['F_thr'][i] = x['F_thr']
    
    return s, x, o, t
