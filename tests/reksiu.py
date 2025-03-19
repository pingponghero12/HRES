import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import math

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from hres.core.sim_loop import sim_loop
from hres.utils.shift_of import shift_of
from hres.utils.impulse import impulse

def load_propellant_data(propellant_name="Paraffin"):
    """
    Load propellant data from JSON file
    
    Parameters:
    -----------
    propellant_name : str
        Name of the propellant to load (without .json extension)
        
    Returns:
    --------
    dict
        Propellant data as dictionary
    """
    # Path to propellant data
    propellant_path = project_root / "hres" / "propellants" / f"{propellant_name}.json"
    
    try:
        with open(propellant_path, 'r') as file:
            data = json.load(file)
        print(f"Loaded propellant data: {propellant_name}")
        
        # Convert lists to numpy arrays where needed
        data['prop_OF'] = np.array(data['prop_OF'])
        data['prop_Pc'] = np.array(data['prop_Pc'])
        data['prop_k'] = np.array(data['prop_k'])
        data['prop_M'] = np.array(data['prop_M'])
        data['prop_T'] = np.array(data['prop_T'])
        data['prop_Reg'] = np.array(data['prop_Reg'])
        
        return data
    
    except FileNotFoundError:
        print(f"Error: Propellant file not found: {propellant_path}")
        return None
    except Exception as e:
        print(f"Error loading propellant data: {str(e)}")
        return None

def run_custom_paraffin_motor_test():
    """
    Integration test using custom Paraffin motor configuration
    """
    print("Running hres integration test with custom Paraffin motor configuration...")
    
    # Load the Paraffin propellant data
    propellant_data = load_propellant_data("Paraffin")
    
    if propellant_data is None:
        print("Failed to load propellant data. Aborting test.")
        return
    
    # Calculate tank volume (cylindrical tank)
    tank_diameter = 0.054  # 54 mm in meters
    tank_length = 0.55     # 550 mm in meters
    tank_volume = math.pi * (tank_diameter/2)**2 * tank_length
    
    # Calculate injector CdA
    injector_diameter = 0.0015  # 1.5 mm in meters
    injector_area = math.pi * (injector_diameter/2)**2
    injector_cd = 0.44
    injector_cda = injector_cd * injector_area
    
    # Initialize simulation parameters with the specified configuration
    s = {
        'dt': 0.01,                # Time step (s)
        'tmax': 10.0,              # Maximum simulation time (s)
        'tburn': 0,                # Burn time (0 = unlimited)
        'Pa': 101325,              # Ambient pressure (Pa) - 1 atm
        
        # Tank parameters
        'tnk_D': tank_diameter,    # Tank diameter (m) - 54 mm
        'tnk_V': tank_volume,      # Tank volume (m³) - calculated
        'tnk_X': tank_length,      # Tank position from datum (m)
        
        # Injector parameters
        'inj_CdA': injector_cda,   # Injector discharge coefficient × area
        'inj_N': 21,               # Number of injectors
        
        # Vent parameters
        'vnt_CdA': 0,              # Vent discharge coefficient × area
        'vnt_S': 0,                # Vent state (0 = closed)
        
        # Chamber parameters
        'cmbr_V': 0,               # Chamber volume (0 = auto-calculate)
        'cmbr_X': 0,               # Chamber position from datum (m)
        
        # Grain parameters
        'grn_L': 0.12,             # Grain length (m) - 120 mm
        'grn_OD': 0.042,           # Grain outer diameter (m) - 42 mm
        
        # Nozzle parameters
        'noz_thrt': 0.0145,        # Nozzle throat diameter (m) - 14.5 mm
        'noz_ER': 4.0,             # Nozzle expansion ratio
        'noz_eff': 0.95,           # Nozzle efficiency - 95%
        'noz_Cd': 1.0,             # Nozzle discharge coefficient
        
        # Combustion parameters
        'cstar_eff': 0.90,         # C-star efficiency - 90%
        
        # Motor parameters
        'mtr_m': 1.0,              # Motor mass without propellants (kg)
        'mtr_cg': 0.2,             # Motor center of gravity position (m)
        
        # Calculation options
        'mp_calc': 1,              # Mass properties calculation flag
    }
    
    # Apply propellant data to simulation parameters
    s.update(propellant_data)
    
    # Override regression parameters with custom values
    s['prop_Reg'] = np.array([0.155, 0.5, 0.0])
    
    # Set regression model to shift_of for realistic simulation
    s['regression_model_func'] = shift_of
    
    print("Motor Configuration:")
    print(f"  Oxidizer Tank: {tank_diameter*1000:.1f}mm diameter × {tank_length*1000:.1f}mm length ({tank_volume*1000000:.0f} cc)")
    # print(f"  Fuel Grain: {s['grn_OD']*1000:.1f}mm OD × {s['grn_ID']*1000:.1f}mm ID × {s['grn_L']*1000:.1f}mm length")
    print(f"  Nozzle: {s['noz_thrt']*1000:.1f}mm throat, {s['noz_ER']:.1f}:1 expansion ratio")
    print(f"  Injector: {injector_diameter*1000:.1f}mm × {s['inj_N']} holes, Cd = {injector_cd:.2f}")
    
    print("\nPropellant properties:")
    print(f"  Name: {s['prop_nm']}")
    print(f"  Density: {s['prop_Rho']} kg/m³")
    print(f"  Custom regression parameters: a={s['prop_Reg'][0]:.3f}, n={s['prop_Reg'][1]:.1f}, m={s['prop_Reg'][2]:.1f}")
    print(f"  C* efficiency: {s['cstar_eff']:.1%}")
    
    # Initialize state variables
    x = {
        'T_tnk': 273.1,            # Tank temperature (K)
        'm_o': 0.85,               # Initial oxidizer mass (kg) - 850g
        'm_f': 0.0,                # Initial fuel mass (kg) - calculated below
        'm_g': 0.001,              # Initial gas mass in chamber (kg)
        'grn_ID': 0.02,            # Grain inner diameter (m) - 20mm
        'grn_ID_old': 0.02,        # Previous grain inner diameter (m)
        'P_cmbr': 101325,          # Initial chamber pressure (Pa) - 1 atm
        'P_tnk': 0,                # Initial tank pressure (Pa)
        'mdot_o': 0,               # Initial oxidizer flow rate (kg/s)
        'mdot_f': 0,               # Initial fuel flow rate (kg/s)
        'mdot_n': 0,               # Initial nozzle flow rate (kg/s)
        'mdot_v': 0,               # Initial vent flow rate (kg/s)
        'mLiq_old': 0.75,          # Initial liquid mass (kg) - estimate
        'mLiq_new': 0.75,          # Current liquid mass (kg)
        'rdot': 0,                 # Regression rate (m/s)
        'OF': 0.0,                 # Initial O/F ratio (will be calculated)
        'F_thr': 0,                # Initial thrust (N)
        'dP': 0                    # Initial pressure change (Pa)
    }
    
    # Calculate initial fuel mass based on grain geometry and density
    grain_volume = math.pi * (s['grn_OD']**2 - x['grn_ID']**2) / 4 * s['grn_L']
    x['m_f'] = grain_volume * s['prop_Rho']
    print(f"  Initial fuel mass: {x['m_f']*1000:.1f}g")
    print(f"  Initial oxidizer mass: {x['m_o']*1000:.1f}g")
    print(f"  Total propellant: {(x['m_f'] + x['m_o'])*1000:.1f}g")
    
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
    print("\nStarting simulation...")
    s, x, o, t = sim_loop(s, x, o, t)
    print(f"Simulation ended at t = {t:.2f} s")
    print(f"End condition: {o['sim_end_cond']}")
    
    # Calculate performance metrics
    total_impulse = np.trapz(o['F_thr'], o['t'])
    initial_propellant_mass = o['m_o'][0] + o['m_f'][0]
    specific_impulse = total_impulse / (9.81 * initial_propellant_mass)
    avg_thrust = np.mean(o['F_thr'][o['F_thr'] > 10])  # Exclude very low thrust
    max_thrust = np.max(o['F_thr'])
    
    # Calculate motor class
    motor_class, percent = impulse(total_impulse)
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"  Total impulse: {total_impulse:.1f} N·s")
    print(f"  Motor class: {motor_class}-{percent:.1f}%")
    print(f"  Specific impulse: {specific_impulse:.1f} s")
    print(f"  Average thrust: {avg_thrust:.1f} N")
    print(f"  Maximum thrust: {max_thrust:.1f} N")
    print(f"  Burn time: {t:.2f} s")
    print(f"  Final grain port diameter: {o['grn_ID'][-1]*1000:.1f} mm")
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Plot 1: Thrust vs Time
    ax = axes[0, 0]
    ax.plot(o['t'], o['F_thr'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.set_title('Thrust vs Time')
    ax.grid(True)
    
    # Plot 2: Pressure vs Time
    ax = axes[0, 1]
    ax.plot(o['t'], o['P_cmbr']/1e6, label='Chamber')
    ax.plot(o['t'], o['P_tnk']/1e6, label='Tank')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_title('Pressure vs Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Mass vs Time
    ax = axes[1, 0]
    ax.plot(o['t'], o['m_o'], label='Oxidizer')
    ax.plot(o['t'], o['m_f'], label='Fuel')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('Propellant Mass vs Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: O/F Ratio vs Time
    ax = axes[1, 1]
    ax.plot(o['t'], o['OF'])
    ax.axhline(y=s['opt_OF'], color='r', linestyle='--', label=f'Optimum ({s["opt_OF"]})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('O/F Ratio')
    ax.set_title('O/F Ratio vs Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 5: Mass Flow Rates
    ax = axes[2, 0]
    ax.plot(o['t'], o['mdot_o'], 'b-', label='Oxidizer')
    ax.plot(o['t'], o['mdot_f'], 'g-', label='Fuel')
    ax.plot(o['t'], o['mdot_n'], 'r-', label='Nozzle')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass Flow Rate (kg/s)')
    ax.set_title('Mass Flow Rates vs Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 6: Regression Rate & Port Diameter
    ax1 = axes[2, 1]
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Regression Rate (mm/s)', color=color)
    ax1.plot(o['t'], o['rdot']*1000, color=color)  # Convert to mm/s
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  # Create second y-axis sharing same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Port Diameter (mm)', color=color)
    ax2.plot(o['t'], o['grn_ID']*1000, color=color)  # Convert to mm
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Regression Rate & Port Diameter')
    
    # Add summary text to figure
    summary_text = (
        f"Motor Configuration:\n"
        f"Propellant: {s['prop_nm']}\n"
        f"Tank: {tank_diameter*1000:.1f}mm × {tank_length*1000:.1f}mm\n"
        # f"Grain: {s['grn_OD']*1000:.1f}mm OD × {s['grn_ID'][0]*1000:.1f}mm ID × {s['grn_L']*1000:.1f}mm\n"
        f"Nozzle: {s['noz_thrt']*1000:.1f}mm throat, {s['noz_ER']:.1f}:1 ratio\n\n"
        f"Performance:\n"
        f"Total Impulse: {total_impulse:.1f} N·s ({motor_class}-{percent:.1f}%)\n"
        f"Burn Time: {t:.2f} s\n"
        f"Avg Thrust: {avg_thrust:.1f} N\n"
        f"Max Thrust: {max_thrust:.1f} N\n"
        f"Specific Impulse: {specific_impulse:.1f} s\n"
        f"End Condition: {o['sim_end_cond']}"
    )
    
    fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Adjust layout
    plt.suptitle(f"Custom Paraffin Hybrid Motor Performance - {s['prop_nm']} with N₂O")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save plot
    output_path = project_root / "plots" / "custom_paraffin_motor_test.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved plot to: {output_path}")
    
    plt.show()
    
    return s, x, o, t, total_impulse

if __name__ == "__main__":
    run_custom_paraffin_motor_test()
