import numpy as np
import matplotlib.pyplot as plt

import sys
import os

from pathlib import Path

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from hres.core.sim_loop import sim_loop
from hres.utils.const_of import const_of
from hres.utils.shift_of import shift_of
from hres.utils.impulse import impulse

def run_hybrid_rocket_simulation():
    """
    Example of setting up and running a complete hres simulation
    """
    # Initialize simulation parameters
    s = {
        'dt': 0.01,                # Time step (s)
        'tmax': 10.0,              # Maximum simulation time (s)
        'tburn': 0,                # Burn time (0 = unlimited)
        'Pa': 101325,              # Ambient pressure (Pa)
        
        # Tank parameters
        'tnk_D': 0.1,              # Tank diameter (m)
        'tnk_V': 0.002,            # Tank volume (m³)
        'tnk_X': 0.3,              # Tank position from datum (m)
        
        # Injector parameters
        'inj_CdA': 2e-6,           # Injector discharge coefficient × area
        'inj_N': 4,                # Number of injectors
        
        # Vent parameters
        'vnt_CdA': 0,              # Vent discharge coefficient × area
        'vnt_S': 0,                # Vent state (0 = closed)
        
        # Chamber parameters
        'cmbr_V': 0,               # Chamber volume (0 = auto-calculate)
        'cmbr_X': 0,               # Chamber position from datum (m)
        
        # Grain parameters
        'grn_L': 0.5,              # Grain length (m)
        'grn_OD': 0.098,           # Grain outer diameter (m)
        
        # Propellant parameters
        'prop_Rho': 1050,          # Propellant density (kg/m³)
        'const_OF': 5.0,           # Target OF ratio for const_OF model
        'prop_Reg': [0.014, 0.5, 0.0],  # Regression coefficients [a, n, m]
        
        # Nozzle parameters
        'noz_thrt': 0.015,         # Nozzle throat diameter (m)
        'noz_ER': 4.0,             # Nozzle expansion ratio
        'noz_eff': 0.95,           # Nozzle efficiency
        'noz_Cd': 0.95,            # Nozzle discharge coefficient
        
        # Combustion parameters
        'cstar_eff': 0.95,         # C-star efficiency
        
        # Motor parameters
        'mtr_m': 1.0,              # Motor mass without propellants (kg)
        'mtr_cg': 0.2,             # Motor center of gravity position (m)
        
        # Calculation options
        'mp_calc': 1,              # Mass properties calculation flag
        
        # Interpolation data for combustion properties
        # These would typically come from a propellant config file
        'prop_OF': np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'prop_Pc': np.array([1000000, 2000000, 3000000, 4000000]),
        'prop_k': np.ones((6, 4)) * 1.2,  # Placeholder data
        'prop_M': np.ones((6, 4)) * 28.0, # Placeholder data
        'prop_T': np.ones((6, 4)) * 3000  # Placeholder data
    }
    
    # Select regression model
    # Point to the desired function
    s['regression_model_func'] = const_of  # or shift_of
    
    # Initialize state variables
    x = {
        'T_tnk': 293.15,           # Tank temperature (K)
        'm_o': 1.8,                # Initial oxidizer mass (kg)
        'm_f': 0.4,                # Initial fuel mass (kg)
        'm_g': 0.001,              # Initial gas mass in chamber (kg)
        'grn_ID': 0.03,            # Grain inner diameter (m)
        'grn_ID_old': 0.03,        # Previous grain inner diameter (m)
        'P_cmbr': 101325,          # Initial chamber pressure (Pa)
        'P_tnk': 0,                # Initial tank pressure (Pa)
        'mdot_o': 0,               # Initial oxidizer flow rate (kg/s)
        'mdot_f': 0,               # Initial fuel flow rate (kg/s)
        'mdot_n': 0,               # Initial nozzle flow rate (kg/s)
        'mdot_v': 0,               # Initial vent flow rate (kg/s)
        'mLiq_old': 1.6,           # Initial liquid mass (kg)
        'mLiq_new': 1.6,           # Current liquid mass (kg)
        'rdot': 0,                 # Regression rate (m/s)
        'OF': 5.0,                 # Initial O/F ratio
        'F_thr': 0,                # Initial thrust (N)
        'dP': 0                    # Initial pressure change (Pa)
    }
    
    # Pre-allocate output arrays
    max_iterations = int(s['tmax'] / s['dt']) + 100  # Add margin
    
    o = {
        't': np.zeros(max_iterations),
        'm_o': np.zeros(max_iterations),
        'm_f': np.zeros(max_iterations),
        'P_tnk': np.zeros(max_iterations),
        'P_cmbr': np.zeros(max_iterations),
        'mdot_o': np.zeros(max_iterations),
        'mdot_f': np.zeros(max_iterations),
        'OF': np.zeros(max_iterations),
        'grn_ID': np.zeros(max_iterations),
        'mdot_n': np.zeros(max_iterations),
        'rdot': np.zeros(max_iterations),
        'dP': np.zeros(max_iterations),
        'F_thr': np.zeros(max_iterations),
        'm_t': np.zeros(max_iterations),
        'cg': np.zeros(max_iterations)
    }
    
    # Initial time
    t = 0.0
    
    # Run simulation
    print("Starting simulation...")
    s, x, o, t = sim_loop(s, x, o, t)
    print(f"Simulation ended at t = {t:.2f} s")
    print(f"End condition: {o['sim_end_cond']}")
    
    # Calculate total impulse
    total_impulse = np.trapz(o['F_thr'], o['t'])
    print(f"Total impulse: {total_impulse:.2f} N·s")
    
    # Get motor class
    motor_class, percent = impulse(total_impulse)
    print(f"Motor class: {motor_class}-{percent:.1f}%")
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Thrust vs Time
    plt.subplot(2, 2, 1)
    plt.plot(o['t'], o['F_thr'])
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.title('Thrust vs Time')
    plt.grid(True)
    
    # Plot 2: Pressure vs Time
    plt.subplot(2, 2, 2)
    plt.plot(o['t'], o['P_cmbr']/1e6, label='Chamber')
    plt.plot(o['t'], o['P_tnk']/1e6, label='Tank')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (MPa)')
    plt.title('Pressure vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Mass vs Time
    plt.subplot(2, 2, 3)
    plt.plot(o['t'], o['m_o'], label='Oxidizer')
    plt.plot(o['t'], o['m_f'], label='Fuel')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.title('Propellant Mass vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: O/F Ratio vs Time
    plt.subplot(2, 2, 4)
    plt.plot(o['t'], o['OF'])
    plt.xlabel('Time (s)')
    plt.ylabel('O/F Ratio')
    plt.title('O/F Ratio vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hres_simulation_results.png', dpi=300)
    plt.show()
    
    return s, x, o, t

if __name__ == "__main__":
    run_hybrid_rocket_simulation()
