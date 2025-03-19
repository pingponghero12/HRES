import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import os

from pathlib import Path

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

# Add the project directory to the path
sys.path.insert(0, os.path.abspath('.'))

from hres.utils.tank import tank
from hres.utils.nox import nox

def test_tank_emptying():
    """
    Test the tank emptying process and visualize the results
    """
    print("Testing tank emptying simulation...")
    
    # Define simulation parameters
    s = {
        'dt': 0.01,                # Time step in seconds
        'tburn': 0,                # Burn time (0 = unlimited)
        'Pa': 101325,              # Ambient pressure (Pa)
        'tnk_V': 0.002,            # Tank volume (m³)
        'inj_CdA': 2e-6,           # Injector discharge coefficient × area
        'inj_N': 4,                # Number of injectors
        'vnt_CdA': 0,              # Vent discharge coefficient × area
        'vnt_S': 0                 # Vent state (0 = closed)
    }
    
    # Initialize state variables
    x = {
        'T_tnk': 293.15,           # Tank temperature (K)
        'm_o': 1.8,                # Initial oxidizer mass (kg)
        'P_cmbr': 2000000,         # Chamber pressure (Pa)
        'mdot_o': 0,               # Oxidizer mass flow rate
        'mdot_v': 0,               # Vent mass flow rate
        'mLiq_old': 1.6,           # Previous liquid mass
        'mLiq_new': 1.6,           # Current liquid mass
        'dP': 0                    # Pressure change
    }
    
    # Get initial ox properties
    x['ox_props'] = nox(x['T_tnk'])
    
    # Calculate initial liquid mass based on tank volume and vapor density
    # Initial liquid mass is the total mass minus vapor mass that fills the tank
    vapor_mass = s['tnk_V'] * x['ox_props']['rho_v'] * 0.1  # 10% vapor initially
    x['mLiq_new'] = x['m_o'] - vapor_mass
    x['mLiq_old'] = x['mLiq_new']
    
    # Initialize output history
    o = {
        't': [0],                  # Time
        'm_o': [x['m_o']],         # Oxidizer mass
        'P_tnk': [0],              # Tank pressure
        'T_tnk': [x['T_tnk']],     # Tank temperature
        'mdot_o': [0],             # Oxidizer mass flow rate
        'mLiq': [x['mLiq_new']],   # Liquid mass
        'dP': [0]                  # Pressure change
    }
    
    # Run simulation until tank is empty
    t = 0
    i = 0
    max_iterations = 10000  # Safety limit
    
    while x['m_o'] > 0.05 and i < max_iterations:  # Run until 95% empty
        # Update tank state
        x = tank(s, o, x, t)
        
        # Record data
        t += s['dt']
        i += 1
        
        o['t'].append(t)
        o['m_o'].append(x['m_o'])
        o['P_tnk'].append(x['P_tnk'])
        o['T_tnk'].append(x['T_tnk'])
        o['mdot_o'].append(x['mdot_o'])
        o['mLiq'].append(x['mLiq_new'])
        o['dP'].append(x.get('dP', 0))
    
    # Convert lists to numpy arrays for plotting
    for key in o:
        o[key] = np.array(o[key])
    
    print(f"Simulation completed in {i} iterations, final time: {t:.2f} seconds")
    print(f"Final oxidizer mass: {x['m_o']:.3f} kg")
    print(f"Final tank temperature: {x['T_tnk']:.2f} K")
    print(f"Final tank pressure: {x['P_tnk']/1e6:.2f} MPa")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Mass vs Time
    plt.subplot(2, 2, 1)
    plt.plot(o['t'], o['m_o'], 'b-', label='Total Mass')
    plt.plot(o['t'], o['mLiq'], 'r--', label='Liquid Mass')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.title('Oxidizer Mass vs Time')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Pressure vs Time
    plt.subplot(2, 2, 2)
    plt.plot(o['t'], o['P_tnk']/1e6)
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (MPa)')
    plt.title('Tank Pressure vs Time')
    plt.grid(True)
    
    # Plot 3: Temperature vs Time
    plt.subplot(2, 2, 3)
    plt.plot(o['t'], o['T_tnk'] - 273.15)  # Convert to Celsius
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Tank Temperature vs Time')
    plt.grid(True)
    
    # Plot 4: Mass Flow Rate vs Time
    plt.subplot(2, 2, 4)
    plt.plot(o['t'][1:], o['mdot_o'][1:])  # Skip first point as it's often 0
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Flow Rate (kg/s)')
    plt.title('Oxidizer Flow Rate vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tank_emptying_test.png', dpi=300)
    plt.show()
    
    return o, x  # Return data for further analysis if needed

if __name__ == "__main__":
    test_tank_emptying()
