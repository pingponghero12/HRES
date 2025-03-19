import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, UTC

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from hres.core.sim_loop import sim_loop
from hres.utils.shift_of import shift_of
from hres.utils.nox import nox

def run_tank_mass_debug():
    """Debug test showing vapor and liquid mass distribution in the tank"""
    print(f"Running tank mass distribution debug - {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: {os.getlogin()}")
    
    # Initialize simulation parameters
    s = {
        'dt': 0.01,                # Time step (s)
        'tmax': 10.0,               # Maximum simulation time (s)
        'tburn': 0,                # Burn time (0 = unlimited)
        'Pa': 101325,              # Ambient pressure (Pa)
        'tnk_D': 0.1,              # Tank diameter (m)
        'tnk_V': 0.002,            # Tank volume (m³)
        'tnk_X': 0.3,              # Tank position from datum (m)
        'inj_CdA': 2e-6,           # Injector discharge coefficient × area
        'inj_N': 4,                # Number of injectors
        'vnt_CdA': 0,              # Vent discharge coefficient × area
        'vnt_S': 0,                # Vent state (0 = closed)
        'cmbr_V': 0,               # Chamber volume (0 = auto-calculate)
        'cmbr_X': 0,               # Chamber position from datum (m)
        'grn_L': 0.2,              # Grain length (m)
        'grn_OD': 0.05,            # Grain outer diameter (m)
        'cmbr_V_head': 0.00005,    # Head-end chamber volume (m³)
        'cmbr_V_aft': 0.00005,     # Aft-end chamber volume (m³)
        'prop_Rho': 920,           # Propellant density (kg/m³)
        'prop_Reg': [0.0986, 0.6518, 0],  # Regression parameters
        'noz_thrt': 0.01,          # Nozzle throat diameter (m)
        'noz_ER': 4.0,             # Nozzle expansion ratio
        'noz_eff': 0.95,           # Nozzle efficiency
        'noz_Cd': 0.95,            # Nozzle discharge coefficient
        'cstar_eff': 0.95,         # C-star efficiency
        'mtr_m': 1.0,              # Motor mass without propellants (kg)
        'mtr_cg': 0.2,             # Motor center of gravity position (m)
        'mp_calc': 1,              # Mass properties calculation flag
        'prop_OF': np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'prop_Pc': np.array([1000000, 2000000, 3000000, 4000000]),
        'prop_k': np.ones((6, 4)) * 1.2,
        'prop_M': np.ones((6, 4)) * 28.0,
        'prop_T': np.ones((6, 4)) * 3000
    }
    
    # Set regression model
    s['regression_model_func'] = shift_of
    
    # Initialize state variables
    x = {
        'T_tnk': 293.15,           # Tank temperature (K)
        'm_o': 1.0,                # Initial oxidizer mass (kg)
        'm_f': 0.1,                # Initial fuel mass (kg)
        'm_g': 0.001,              # Initial gas mass in chamber (kg)
        'grn_ID': 0.02,            # Grain inner diameter (m)
        'grn_ID_old': 0.02,        # Previous grain inner diameter (m)
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
    max_iterations = int(s['tmax'] / s['dt']) + 100
    
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
        'cg': np.zeros(max_iterations),
        'T_tnk': np.zeros(max_iterations),   # Tank temperature
        'mLiq': np.zeros(max_iterations),    # Liquid mass
        'mVap': np.zeros(max_iterations),    # Vapor mass
        'rho_l': np.zeros(max_iterations),   # Liquid density
        'rho_v': np.zeros(max_iterations),   # Vapor density
        'Vl': np.zeros(max_iterations),      # Liquid volume
        'Vv': np.zeros(max_iterations),      # Vapor volume
        'Z': np.zeros(max_iterations)        # Compressibility factor
    }
    
    # Custom simulation loop with extra tracking for tank properties
    from hres.core.sim_iteration import sim_iteration
    
    print("Starting simulation...")
    
    i = 0
    t = 0.0
    
    while True:
        # Pre-iteration - capture state at the start of this step
        if i < max_iterations:
            o['t'][i] = t
            o['T_tnk'][i] = x['T_tnk']
            o['mLiq'][i] = x['mLiq_new']
            o['m_o'][i] = x['m_o']
            
            # Get oxidizer properties
            ox_props = nox(x['T_tnk'])
            
            # Calculate vapor mass
            o['mVap'][i] = x['m_o'] - x['mLiq_new']
            
            # Record densities
            o['rho_l'][i] = ox_props['rho_l']
            o['rho_v'][i] = ox_props['rho_v']
            
            # Calculate volumes
            if x['mLiq_new'] > 0:
                o['Vl'][i] = x['mLiq_new'] / ox_props['rho_l']
            else:
                o['Vl'][i] = 0
                
            o['Vv'][i] = (x['m_o'] - x['mLiq_new']) / ox_props['rho_v'] if ox_props['rho_v'] > 0 else 0
            
            # Compressibility factor
            o['Z'][i] = ox_props['Z']
        
        # Run simulation iteration
        s, x, o, t = sim_iteration(s, x, o, t, i)
        i += 1
        
        # Check for end conditions
        if i >= max_iterations or t >= s['tmax'] or x['m_o'] <= 0 or x['grn_ID'] >= s['grn_OD']:
            break
    
    # Trim arrays to actual number of iterations
    for key in o:
        o[key] = o[key][:i]
    
    print(f"Simulation completed: {i} iterations over {t:.2f} seconds")
    
    # Calculate ullage (vapor volume fraction)
    ullage = o['Vv'] / s['tnk_V'] * 100  # in percent
    
    # Create diagnostic plots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Tank Mass Distribution
    plt.subplot(3, 2, 1)
    plt.plot(o['t'], o['mLiq'], 'b-', label='Liquid')
    plt.plot(o['t'], o['mVap'], 'r-', label='Vapor')
    plt.plot(o['t'], o['m_o'], 'k--', label='Total')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.title('Tank Mass Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Tank Volume Distribution
    plt.subplot(3, 2, 2)
    plt.plot(o['t'], o['Vl']*1e6, 'b-', label='Liquid')
    plt.plot(o['t'], o['Vv']*1e6, 'r-', label='Vapor')
    plt.plot(o['t'], np.ones_like(o['t'])*s['tnk_V']*1e6, 'k--', label='Tank')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (cm³)')
    plt.title('Tank Volume Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Tank Temperature
    plt.subplot(3, 2, 3)
    plt.plot(o['t'], o['T_tnk'])
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Tank Temperature vs Time')
    plt.grid(True)
    
    # Plot 4: Ullage Percentage
    plt.subplot(3, 2, 4)
    plt.plot(o['t'], ullage)
    plt.xlabel('Time (s)')
    plt.ylabel('Ullage (%)')
    plt.title('Tank Ullage vs Time')
    plt.grid(True)
    
    # Plot 5: Pressures
    plt.subplot(3, 2, 5)
    plt.plot(o['t'], o['P_tnk']/1e6, 'r-', label='Tank')
    plt.plot(o['t'], o['P_cmbr']/1e6, 'b-', label='Chamber')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (MPa)')
    plt.title('Pressure vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Mass Flow Rates
    plt.subplot(3, 2, 6)
    plt.plot(o['t'], o['mdot_o'], 'b-', label='Oxidizer')
    plt.plot(o['t'], o['mdot_f'], 'g-', label='Fuel')
    plt.plot(o['t'], o['mdot_n'], 'r-', label='Nozzle')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Flow Rate (kg/s)')
    plt.title('Mass Flow Rates vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(project_root / 'plots' / 'tank_mass_debug.png'), dpi=300)
    print(f"Saved plot to {project_root / 'plots' / 'tank_mass_debug.png'}")
    
    # Create a second set of plots focusing on densities and compressibility
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Densities
    plt.subplot(2, 2, 1)
    plt.plot(o['t'], o['rho_l'], 'b-', label='Liquid')
    plt.plot(o['t'], o['rho_v'], 'r-', label='Vapor')
    plt.xlabel('Time (s)')
    plt.ylabel('Density (kg/m³)')
    plt.title('Phase Densities vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Compressibility Factor
    plt.subplot(2, 2, 2)
    plt.plot(o['t'], o['Z'])
    plt.xlabel('Time (s)')
    plt.ylabel('Z Factor')
    plt.title('Compressibility Factor vs Time')
    plt.grid(True)
    
    # Plot 3: Phase Diagram - Temperature vs Pressure
    plt.subplot(2, 2, 3)
    plt.plot(o['T_tnk'], o['P_tnk']/1e6, 'r-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (MPa)')
    plt.title('N₂O Phase Diagram Trajectory')
    plt.grid(True)
    
    # Plot 4: Liquid and Vapor Mass with Temperature
    ax1 = plt.subplot(2, 2, 4)
    ax1.plot(o['t'], o['mLiq'], 'b-', label='Liquid')
    ax1.plot(o['t'], o['mVap'], 'r-', label='Vapor')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mass (kg)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_title('Mass Distribution and Temperature')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(o['t'], o['T_tnk'], 'g--', label='Temperature')
    ax2.set_ylabel('Temperature (K)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(str(project_root / 'plots' / 'tank_phase_debug.png'), dpi=300)
    print(f"Saved plot to {project_root / 'plots' / 'tank_phase_debug.png'}")
    
    plt.show()
    
    # Print summary of key values
    print("\nTank Mass/Volume Summary:")
    print(f"  Initial liquid mass: {o['mLiq'][0]:.3f} kg")
    print(f"  Initial vapor mass: {o['mVap'][0]:.3f} kg")
    print(f"  Initial temperature: {o['T_tnk'][0]:.2f} K")
    print(f"  Initial pressure: {o['P_tnk'][0]/1e6:.3f} MPa")
    print(f"  Initial liquid volume: {o['Vl'][0]*1e6:.1f} cm³")
    print(f"  Initial vapor volume: {o['Vv'][0]*1e6:.1f} cm³")
    print(f"  Initial ullage: {ullage[0]:.1f}%")
    
    if len(o['t']) > 1:
        print("\nChange rates during liquid phase:")
        # Find index where liquid is depleted
        liquid_end_idx = np.argmin(o['mLiq'] > 0) if np.any(o['mLiq'] == 0) else len(o['t'])
        if liquid_end_idx > 10:  # Need some data points to calculate rates
            # Calculate rates during liquid phase
            t_liq = o['t'][:liquid_end_idx]
            T_liq = o['T_tnk'][:liquid_end_idx]
            P_liq = o['P_tnk'][:liquid_end_idx]
            
            dT_dt = (T_liq[-1] - T_liq[10]) / (t_liq[-1] - t_liq[10])  # Skip first few points
            dP_dt = (P_liq[-1] - P_liq[10]) / (t_liq[-1] - t_liq[10])
            
            print(f"  Temperature change rate: {dT_dt:.3f} K/s")
            print(f"  Pressure change rate: {dP_dt/1e6:.6f} MPa/s")
    
    return s, x, o, t

if __name__ == "__main__":
    run_tank_mass_debug()
