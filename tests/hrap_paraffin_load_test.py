import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

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

def run_paraffin_integration_test():
    """
    Integration test using Paraffin propellant data
    """
    print("Running hres integration test with Paraffin propellant...")
    
    # Load the Paraffin propellant data
    propellant_data = load_propellant_data("Paraffin")
    
    if propellant_data is None:
        print("Failed to load propellant data. Aborting test.")
        return
    
    # Initialize simulation parameters
    s = {
        'dt': 0.01,                # Time step (s)
        'tmax': 5.0,               # Maximum simulation time (s)
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
        'grn_L': 0.25,             # Grain length (m)
        'grn_OD': 0.08,            # Grain outer diameter (m)
        
        # Nozzle parameters
        'noz_thrt': 0.012,         # Nozzle throat diameter (m)
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
    }
    
    # Apply propellant data to simulation parameters
    s.update(propellant_data)
    
    # Set regression model to shift_of for realistic simulation
    s['regression_model_func'] = shift_of
    
    print("Propellant properties:")
    print(f"  Name: {s['prop_nm']}")
    print(f"  Density: {s['prop_Rho']} kg/m³")
    print(f"  Regression parameters: a={s['prop_Reg'][0]:.4f}, n={s['prop_Reg'][1]:.4f}, m={s['prop_Reg'][2]:.4f}")
    print(f"  Optimum O/F ratio: {s['opt_OF']:.2f}")
    
    # Initialize state variables
    x = {
        'T_tnk': 293.15,           # Tank temperature (K)
        'm_o': 1.0,                # Initial oxidizer mass (kg)
        'm_f': 0.18,               # Initial fuel mass (kg)
        'm_g': 0.001,              # Initial gas mass in chamber (kg)
        'grn_ID': 0.03,            # Grain inner diameter (m)
        'grn_ID_old': 0.03,        # Previous grain inner diameter (m)
        'P_cmbr': 101325,          # Initial chamber pressure (Pa)
        'P_tnk': 0,                # Initial tank pressure (Pa)
        'mdot_o': 0,               # Initial oxidizer flow rate (kg/s)
        'mdot_f': 0,               # Initial fuel flow rate (kg/s)
        'mdot_n': 0,               # Initial nozzle flow rate (kg/s)
        'mdot_v': 0,               # Initial vent flow rate (kg/s)
        'mLiq_old': 0.9,           # Initial liquid mass (kg)
        'mLiq_new': 0.9,           # Current liquid mass (kg)
        'rdot': 0,                 # Regression rate (m/s)
        'OF': 0.0,                 # Initial O/F ratio (will be calculated)
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
    print("Starting integrated simulation...")
    s, x, o, t = sim_loop(s, x, o, t)
    print(f"Simulation ended at t = {t:.2f} s")
    print(f"End condition: {o['sim_end_cond']}")
    
    # Calculate total impulse
    total_impulse = np.trapz(o['F_thr'], o['t'])
    print(f"Total impulse: {total_impulse:.2f} N·s")
    
    # Get motor class
    motor_class, percent = impulse(total_impulse)
    print(f"Motor class: {motor_class}-{percent:.1f}%")
    
    # Calculate performance metrics
    avg_thrust = np.mean(o['F_thr'][o['F_thr'] > 0])
    max_thrust = np.max(o['F_thr'])
    burn_time = t
    avg_chamber_pressure = np.mean(o['P_cmbr'][o['P_cmbr'] > s['Pa']])
    propellant_mass = x['m_o'] + x['m_f']
    specific_impulse = total_impulse / (9.81 * propellant_mass)
    
    print("\nPerformance Summary:")
    print(f"  Average thrust: {avg_thrust:.2f} N")
    print(f"  Maximum thrust: {max_thrust:.2f} N")
    print(f"  Burn time: {burn_time:.2f} s")
    print(f"  Average chamber pressure: {avg_chamber_pressure/1e6:.2f} MPa")
    print(f"  Specific impulse: {specific_impulse:.0f} s")
    
    # Create plots
    print("\nGenerating performance plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
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
    
    # Plot 5: Regression Rate vs Time
    ax = axes[2, 0]
    ax.plot(o['t'], o['rdot']*1000)  # Convert to mm/s
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Regression Rate (mm/s)')
    ax.set_title('Fuel Regression Rate vs Time')
    ax.grid(True)
    
    # Plot 6: Port Diameter vs Time
    ax = axes[2, 1]
    ax.plot(o['t'], o['grn_ID']*1000)  # Convert to mm
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Port Diameter (mm)')
    ax.set_title('Fuel Grain Port Diameter vs Time')
    ax.grid(True)
    
    plt.suptitle(f"hres Simulation Results - {s['prop_nm']} Propellant")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plot_path = project_root / "plots" / "hres_integration_test_results.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    
    plt.show()
    
    return s, x, o, t, total_impulse

def test_paraffin_expected_performance():
    """
    Test that the Paraffin propellant performance meets expected values
    """
    s, x, o, t, total_impulse = run_paraffin_integration_test()
    
    # Calculate key performance metrics
    avg_thrust = np.mean(o['F_thr'][o['F_thr'] > 0])
    max_thrust = np.max(o['F_thr'])
    burn_time = t
    propellant_mass = x['m_o'] + x['m_f']
    specific_impulse = total_impulse / (9.81 * propellant_mass)
    
    # Define expected performance ranges
    # These would be adjusted based on known Paraffin/N2O performance
    expected_ranges = {
        'total_impulse': (400, 1500),  # N·s
        'avg_thrust': (100, 500),      # N
        'max_thrust': (200, 700),      # N
        'burn_time': (1.5, 7),         # s
        'specific_impulse': (150, 250)  # s
    }
    
    # Check if performance is within expected ranges
    checks = {
        'total_impulse': expected_ranges['total_impulse'][0] <= total_impulse <= expected_ranges['total_impulse'][1],
        'avg_thrust': expected_ranges['avg_thrust'][0] <= avg_thrust <= expected_ranges['avg_thrust'][1],
        'max_thrust': expected_ranges['max_thrust'][0] <= max_thrust <= expected_ranges['max_thrust'][1],
        'burn_time': expected_ranges['burn_time'][0] <= burn_time <= expected_ranges['burn_time'][1],
        'specific_impulse': expected_ranges['specific_impulse'][0] <= specific_impulse <= expected_ranges['specific_impulse'][1]
    }
    
    # Print results with pass/fail indicators
    print("\nPerformance Checks:")
    for param, is_valid in checks.items():
        status = "PASS" if is_valid else "FAIL"
        print(f"  {param}: {status}")
    
    # Overall test result
    all_passed = all(checks.values())
    print(f"\nIntegration Test: {'PASS' if all_passed else 'FAIL'}")
    
    # For automated testing, return the result
    return all_passed

if __name__ == "__main__":
    # For interactive use, just run the integration test
    run_paraffin_integration_test()
    
    # For automated testing, uncomment:
    # test_result = test_paraffin_expected_performance()
    # sys.exit(0 if test_result else 1)
