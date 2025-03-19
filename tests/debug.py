import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from hres.core.sim_loop import sim_loop
from hres.utils.shift_of import shift_of
from hres.utils.nox import nox

def run_diagnostic_test(save_plots=True):
    """
    Run a detailed diagnostic test to investigate chamber pressure behavior issues
    
    Parameters:
    -----------
    save_plots : bool
        Whether to save the diagnostic plots to files
    
    Returns:
    --------
    tuple
        (s, x, o, debug) - Simulation parameters, final state, outputs, and debug data
    """
    print(f"Running hres pressure diagnostic test - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: {os.getlogin()}")
    
    # Initialize simulation parameters - Use a simplified test case
    s = {
        'dt': 0.01,                # Time step (s)
        'tmax': 8.0,               # Maximum simulation time (s)
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
        'grn_L': 0.2,              # Grain length (m)
        'grn_OD': 0.05,            # Grain outer diameter (m)
        
        # Propellant parameters
        'prop_Rho': 920,           # Propellant density (kg/m³) - Paraffin
        'prop_Reg': [0.0986, 0.6518, 0],  # Regression parameters for Paraffin
        
        # Nozzle parameters
        'noz_thrt': 0.01,          # Nozzle throat diameter (m)
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
        'prop_OF': np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'prop_Pc': np.array([1000000, 2000000, 3000000, 4000000]),
        'prop_k': np.ones((6, 4)) * 1.2,  # Specific heat ratio
        'prop_M': np.ones((6, 4)) * 28.0, # Molecular mass (g/mol)
        'prop_T': np.ones((6, 4)) * 3000  # Temperature (K)
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
    
    # Debug data for diagnostics
    debug = {
        'liquid_fraction': np.zeros(max_iterations),
        'chamber_volume': np.zeros(max_iterations),
        'chamber_dV': np.zeros(max_iterations),
        'chamber_dm_g': np.zeros(max_iterations),
        'chamber_pressure_ratio': np.zeros(max_iterations),
        'dp_calc': np.zeros(max_iterations),
        'tank_temp': np.zeros(max_iterations),
        'pressure_diff': np.zeros(max_iterations),
        'ox_density_v': np.zeros(max_iterations),
        'ox_density_l': np.zeros(max_iterations),
        'chamber_dP': np.zeros(max_iterations)
    }
    
    # Initial time
    t = 0.0
    
    # Custom simulation loop with instrumentation
    from hres.core.sim_iteration import sim_iteration
    
    print("Starting diagnostic simulation...")
    
    i = 0
    while True:
        # Update time and iteration counter
        t = i * s['dt']
        
        # Get oxidizer properties before the step
        ox_props = nox(x['T_tnk'])
        
        # Record pre-step debug information
        if i < max_iterations:
            debug['liquid_fraction'][i] = x['mLiq_new'] / x['m_o'] if x['m_o'] > 0 else 0
            debug['tank_temp'][i] = x['T_tnk']
            debug['pressure_diff'][i] = x['P_tnk'] - x['P_cmbr'] if 'P_tnk' in x else 0
            debug['ox_density_v'][i] = ox_props['rho_v']
            debug['ox_density_l'][i] = ox_props['rho_l']
            
            # Calculate chamber volume
            if s['cmbr_V'] == 0:
                chamber_volume = 0.25 * np.pi * x['grn_ID']**2 * s['grn_L']
            else:
                chamber_volume = s['cmbr_V'] - 0.25 * np.pi * (s['grn_OD']**2 - x['grn_ID']**2) * s['grn_L']
            debug['chamber_volume'][i] = chamber_volume
            
            # Calculate volume change rate
            if i > 0:
                dV = 0.25 * np.pi * (x['grn_ID']**2 - x['grn_ID_old']**2) * s['grn_L'] / s['dt']
                debug['chamber_dV'][i] = dV
        
        # Run one simulation iteration
        s, x, o, t = sim_iteration(s, x, o, t, i)
        i += 1
        
        # Record post-step debug information
        if i < max_iterations:
            # Mass flow balance
            dm_g = x['mdot_f'] + x['mdot_o'] - x['mdot_n']
            debug['chamber_dm_g'][i] = dm_g
            
            # Pressure ratio (chamber/tank)
            debug['chamber_pressure_ratio'][i] = x['P_cmbr'] / x['P_tnk'] if x['P_tnk'] > 0 else 0
            
            # Recreate chamber pressure calculation
            if s['cmbr_V'] == 0:
                V = 0.25 * np.pi * x['grn_ID']**2 * s['grn_L']
            else:
                V = s['cmbr_V'] - 0.25 * np.pi * (s['grn_OD']**2 - x['grn_ID']**2) * s['grn_L']
            
            if x['m_g'] > 0 and V > 0:
                dV = 0.25 * np.pi * (x['grn_ID']**2 - x['grn_ID_old']**2) * s['grn_L'] / s['dt']
                dP = x['P_cmbr'] * (dm_g / x['m_g'] - dV / V)
                debug['chamber_dP'][i] = dP
                
                # Pressure contribution from mass change and volume change
                dp_calc = x['P_cmbr'] * (dm_g / x['m_g'] - dV / V) * s['dt']
                debug['dp_calc'][i] = dp_calc
        
        # Check for simulation end conditions
        if x['grn_ID'] >= s['grn_OD']:
            print("Simulation ended: Fuel Depleted")
            break
        elif x['m_o'] <= 0:
            print("Simulation ended: Oxidizer Depleted")
            break
        elif t >= s['tmax']:
            print("Simulation ended: Max Simulation Time Reached")
            break
        elif x['P_cmbr'] <= s['Pa']:
            print("Simulation ended: Burn Complete")
            break
    
    # Trim output arrays to actual data length
    valid_length = i
    
    for key in o:
        o[key] = o[key][:valid_length]
    
    for key in debug:
        debug[key] = debug[key][:valid_length]
    
    print(f"Simulation completed: {valid_length} iterations over {t:.2f} seconds")
    
    # Create diagnostic plots
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle("hres Pressure Diagnostic Analysis", fontsize=16)
    
    # Plot 1: Pressure vs Time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(o['t'], o['P_cmbr']/1e6, 'b-', label='Chamber')
    ax1.plot(o['t'], o['P_tnk']/1e6, 'r-', label='Tank')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_title('Chamber and Tank Pressure vs Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Mass Flow Rates
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(o['t'], o['mdot_o'], 'b-', label='Oxidizer')
    ax2.plot(o['t'], o['mdot_f'], 'g-', label='Fuel')
    ax2.plot(o['t'], o['mdot_n'], 'r-', label='Nozzle')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mass Flow Rate (kg/s)')
    ax2.set_title('Mass Flow Rates vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Mass Balance and Gas Mass
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(o['t'], debug['chamber_dm_g'], 'b-', label='dm_g')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass Flow Balance (kg/s)')
    ax3.set_title('Chamber Mass Flow Balance (dm_g)')
    ax3.grid(True)
    
    # Plot 4: Chamber Volume Change
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(o['t'], debug['chamber_volume']*1e6, 'b-', label='Volume')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Chamber Volume (cm³)')
    ax4.set_title('Chamber Volume vs Time')
    ax4.grid(True)
    
    # Plot 5: dV and Volume Change Rate
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(o['t'], debug['chamber_dV']*1e6, 'g-', label='dV/dt')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Volume Change Rate (cm³/s)')
    ax5.set_title('Chamber Volume Change Rate')
    ax5.grid(True)
    
    # Plot 6: Pressure Ratio
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(o['t'], debug['chamber_pressure_ratio'], 'b-')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Pressure Ratio (Pc/Pt)')
    ax6.set_title('Chamber to Tank Pressure Ratio')
    ax6.grid(True)
    
    # Plot 7: Liquid Fraction
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(o['t'], debug['liquid_fraction']*100, 'b-')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Liquid Fraction (%)')
    ax7.set_title('Oxidizer Liquid Fraction')
    ax7.grid(True)
    
    # Plot 8: Tank Temperature
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(o['t'], debug['tank_temp'], 'r-')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Temperature (K)')
    ax8.set_title('Tank Temperature vs Time')
    ax8.grid(True)
    
    # Plot 9: Calculated dP
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(o['t'], debug['chamber_dP']/1e6, 'g-')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('dP (MPa/s)')
    ax9.set_title('Chamber Pressure Change Rate')
    ax9.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_plots:
        plot_path = project_root / "plots" / "pressure_diagnostics.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Diagnostic plot saved to: {plot_path}")
    
    # Create a second figure with additional detailed analysis
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle("hres Pressure Change Analysis", fontsize=16)
    
    # Plot 1: Pressure Differential
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(o['t'], debug['pressure_diff']/1e6, 'b-')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pressure Differential (MPa)')
    ax1.set_title('Tank-Chamber Pressure Differential')
    ax1.grid(True)
    
    # Plot 2: Oxidizer Density
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(o['t'], debug['ox_density_l'], 'b-', label='Liquid')
    ax2.plot(o['t'], debug['ox_density_v'], 'r-', label='Vapor')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Density (kg/m³)')
    ax2.set_title('Oxidizer Density vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Calculated dP * dt
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(o['t'], debug['dp_calc']/1e6, 'g-')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('ΔP per step (MPa)')
    ax3.set_title('Chamber Pressure Change per Timestep')
    ax3.grid(True)
    
    # Plot 4: Chamber Pressure and Mass Flows Together
    ax4 = plt.subplot(2, 3, 4)
    ax4_twin = ax4.twinx()
    ln1 = ax4.plot(o['t'], o['P_cmbr']/1e6, 'b-', label='Chamber Pressure')
    ln2 = ax4_twin.plot(o['t'], o['mdot_o'], 'r-', label='Oxidizer Flow')
    ln3 = ax4_twin.plot(o['t'], o['mdot_f'], 'g-', label='Fuel Flow')
    ln4 = ax4_twin.plot(o['t'], o['mdot_n'], 'k-', label='Nozzle Flow')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pressure (MPa)')
    ax4_twin.set_ylabel('Mass Flow (kg/s)')
    ax4.set_title('Chamber Pressure and Mass Flow Rates')
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs, loc='best')
    ax4.grid(True)
    
    # Plot 5: Pressure Change Components
    ax5 = plt.subplot(2, 3, 5)
    # Calculate components that contribute to dP
    dm_g_component = np.zeros_like(o['t'])
    dv_component = np.zeros_like(o['t'])
    valid_indices = np.logical_and(debug['chamber_dm_g'] != 0, debug['chamber_dV'] != 0)
    
    for i in range(len(o['t'])):
        if i > 0 and x['m_g'] > 0 and debug['chamber_volume'][i] > 0:
            dm_g_component[i] = o['P_cmbr'][i] * (debug['chamber_dm_g'][i] / x['m_g']) * s['dt'] / 1e6
            dv_component[i] = -o['P_cmbr'][i] * (debug['chamber_dV'][i] / debug['chamber_volume'][i]) * s['dt'] / 1e6
    
    ax5.plot(o['t'], dm_g_component, 'g-', label='Mass Flow Component')
    ax5.plot(o['t'], dv_component, 'r-', label='Volume Change Component')
    ax5.plot(o['t'], dm_g_component + dv_component, 'b-', label='Total Change')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pressure Change Contribution (MPa)')
    ax5.set_title('Components of Chamber Pressure Change')
    ax5.legend()
    ax5.grid(True)
    
    # Plot 6: Cumulative Pressure Change
    ax6 = plt.subplot(2, 3, 6)
    # Calculate cumulative pressure change
    cumulative_dp = np.cumsum(debug['dp_calc']) / 1e6
    ax6.plot(o['t'], cumulative_dp, 'b-', label='Cumulative ΔP')
    ax6.plot(o['t'], o['P_cmbr']/1e6 - o['P_cmbr'][0]/1e6, 'r--', label='Actual Chamber P Change')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Cumulative Pressure Change (MPa)')
    ax6.set_title('Cumulative Chamber Pressure Change')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_plots:
        plot_path = project_root / "plots" / "pressure_diagnostics_detail.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Detailed diagnostic plot saved to: {plot_path}")
    
    # Create a third figure focusing on key relationships during the liquid phase
    # First, identify when liquid phase ends
    liquid_end_idx = np.argmin(debug['liquid_fraction'] > 0) if np.any(debug['liquid_fraction'] == 0) else len(o['t'])
    
    if liquid_end_idx > 0:
        fig3 = plt.figure(figsize=(18, 10))
        fig3.suptitle("hres Liquid Phase Analysis", fontsize=16)
        
        # Plot 1: Liquid Phase Pressure
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(o['t'][:liquid_end_idx], o['P_cmbr'][:liquid_end_idx]/1e6, 'b-', label='Chamber')
        ax1.plot(o['t'][:liquid_end_idx], o['P_tnk'][:liquid_end_idx]/1e6, 'r-', label='Tank')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pressure (MPa)')
        ax1.set_title('Liquid Phase Pressures')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Liquid Phase Mass Flows
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(o['t'][:liquid_end_idx], o['mdot_o'][:liquid_end_idx], 'b-', label='Oxidizer')
        ax2.plot(o['t'][:liquid_end_idx], o['mdot_f'][:liquid_end_idx], 'g-', label='Fuel')
        ax2.plot(o['t'][:liquid_end_idx], o['mdot_n'][:liquid_end_idx], 'r-', label='Nozzle')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mass Flow Rate (kg/s)')
        ax2.set_title('Liquid Phase Mass Flow Rates')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Liquid Amount and Tank Temperature
        ax3 = plt.subplot(2, 2, 3)
        ax3_twin = ax3.twinx()
        ln1 = ax3.plot(o['t'][:liquid_end_idx], debug['liquid_fraction'][:liquid_end_idx]*100, 'b-', label='Liquid %')
        ln2 = ax3_twin.plot(o['t'][:liquid_end_idx], debug['tank_temp'][:liquid_end_idx], 'r-', label='Tank Temp')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Liquid Fraction (%)')
        ax3_twin.set_ylabel('Temperature (K)')
        ax3.set_title('Liquid Fraction and Tank Temperature')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc='best')
        ax3.grid(True)
        
        # Plot 4: Pressure Change During Liquid Phase
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(o['t'][:liquid_end_idx], debug['chamber_dP'][:liquid_end_idx]/1e6, 'g-')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('dP (MPa/s)')
        ax4.set_title('Chamber Pressure Change Rate (Liquid Phase)')
        ax4.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_plots:
            plot_path = project_root / "plots" / "liquid_phase_diagnostics.png"
            plt.savefig(plot_path, dpi=300)
            print(f"Liquid phase diagnostic plot saved to: {plot_path}")
    
    # Show the plots if running interactively
    plt.show()
    
    return s, x, o, debug

if __name__ == "__main__":
    run_diagnostic_test()
