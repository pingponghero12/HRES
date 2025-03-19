import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize

from pathlib import Path

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from hres.utils.chamber import chamber
from hres.utils.nozzle import nozzle
from hres.utils.shift_of import shift_of
from hres.utils.comb import comb
from hres.utils.interpolation import interp2x

def test_chamber():
    print("Testing chamber...")
    
    # Create test simulation parameters
    s = {
        'dt': 0.01,                # time step in seconds
        'cmbr_V': 0,               # chamber volume (0 means use calculated volume)
        'grn_L': 0.5,              # grain length in meters
        'grn_OD': 0.098,           # grain outer diameter in meters
        'noz_Cd': 0.95,            # nozzle discharge coefficient
        'noz_thrt': 0.02,          # nozzle throat diameter in meters
        'Pa': 101325               # ambient pressure in Pa
    }
    
    # Create test state
    x = {
        'grn_ID': 0.05,            # grain inner diameter in meters
        'grn_ID_old': 0.048,       # previous grain inner diameter
        'P_cmbr': 2000000,         # chamber pressure in Pa
        'mdot_f': 0.02,            # fuel mass flow rate
        'mdot_o': 0.08,            # oxidizer mass flow rate
        'm_g': 0.01,               # mass of gas in chamber
        'cstar': 1500              # characteristic velocity
    }
    
    # Call chamber function
    x_out = chamber(s, x)
    
    # Check that output contains expected fields
    print(f"  mdot_n: {x_out['mdot_n']:.4f} kg/s")
    print(f"  P_cmbr: {x_out['P_cmbr']:.2f} Pa")
    
    # Basic validation
    assert x_out['mdot_n'] > 0, "Nozzle mass flow should be positive"
    assert x_out['P_cmbr'] > s['Pa'], "Chamber pressure should exceed ambient"
    
    # Test with zero oxidizer flow to check end-of-burn condition
    x['mdot_o'] = 0
    x_out = chamber(s, x)
    print(f"  P_cmbr at no oxidizer flow: {x_out['P_cmbr']:.2f} Pa")
    
    print("  chamber test passed!")


def test_nozzle():
    print("Testing nozzle...")
    
    # Create test simulation parameters
    s = {
        'noz_ER': 4.0,             # nozzle expansion ratio (exit area / throat area)
        'noz_eff': 0.9,            # nozzle efficiency
        'noz_Cd': 0.95,            # nozzle discharge coefficient
        'noz_thrt': 0.02,          # nozzle throat diameter in meters
        'Pa': 101325               # ambient pressure in Pa
    }
    
    # Create test state at high chamber pressure
    x = {
        'P_cmbr': 2000000,         # chamber pressure in Pa
        'k': 1.2,                  # specific heat ratio
    }
    
    # Call nozzle function
    x_out = nozzle(s, x)
    
    # Print results
    print(f"  F_thr at high pressure: {x_out['F_thr']:.2f} N")
    
    # Test with chamber pressure equal to ambient
    x['P_cmbr'] = s['Pa']
    x_out = nozzle(s, x)
    print(f"  F_thr at ambient pressure: {x_out['F_thr']:.2f} N")
    
    # Verify thrust is zero when chamber pressure equals ambient
    assert x_out['F_thr'] == 0, "Thrust should be zero when P_cmbr equals Pa"
    
    print("  nozzle test passed!")


def test_shift_of():
    print("Testing shift_of...")
    
    # Create test simulation parameters
    s = {
        'dt': 0.01,                # time step in seconds
        'prop_Rho': 1050.0,        # propellant density in kg/m³
        'grn_L': 0.5,              # grain length in meters
        'prop_Reg': [0.014, 0.5, 0.0]  # regression coefficients [a, n, m]
    }
    
    # Create test state
    x = {
        'mdot_o': 0.08,            # oxidizer mass flow rate in kg/s
        'grn_ID': 0.05,            # grain inner diameter in meters
        'm_f': 1.0                 # fuel mass in kg
    }
    
    # Call shift_of function
    x_out = shift_of(s, x)
    
    # Check results
    print(f"  rdot: {x_out['rdot']:.6f} m/s")
    print(f"  mdot_f: {x_out['mdot_f']:.4f} kg/s")
    print(f"  OF: {x_out['OF']:.2f}")
    print(f"  grn_ID new: {x_out['grn_ID']:.6f} m")
    
    # Basic validation
    port_area = np.pi * (x['grn_ID']/2)**2
    flux = x['mdot_o'] / port_area
    expected_rdot = 0.001 * s['prop_Reg'][0] * flux**s['prop_Reg'][1] * s['grn_L']**s['prop_Reg'][2]
    assert abs(x_out['rdot'] - expected_rdot) < 1e-6, "Regression rate calculation error"
    
    # Test with zero oxidizer flow
    x['mdot_o'] = 0
    x_out = shift_of(s, x)
    print(f"  OF with zero oxidizer flow: {x_out['OF']}")
    assert x_out['OF'] == 0, "OF should be zero when mdot_o is zero"
    
    print("  shift_of test passed!")


def test_comb():
    print("Testing comb...")
    
    # Setup mock interpolation data similar to what would come from a propellant config
    s = {
        'prop_OF': np.array([2.0, 3.0, 4.0, 5.0]),
        'prop_Pc': np.array([1000000, 2000000, 3000000]),
        'prop_k': np.array([[1.2, 1.21, 1.22],
                            [1.18, 1.19, 1.2], 
                            [1.16, 1.17, 1.18],
                            [1.15, 1.16, 1.17]]),
        'prop_M': np.array([[28.0, 28.1, 28.2],
                            [29.0, 29.1, 29.2],
                            [30.0, 30.1, 30.2],
                            [31.0, 31.1, 31.2]]),
        'prop_T': np.array([[2500, 2550, 2600],
                            [2700, 2750, 2800],
                            [2900, 2950, 3000],
                            [3100, 3150, 3200]]),
        'cstar_eff': 0.95,
        'tburn': 5.0                # burn time in seconds
    }
    
    # Mock current state
    x = {
        'OF': 3.5,                 # O/F ratio
        'P_cmbr': 2500000          # chamber pressure in Pa
    }
    
    # Test at t < tburn
    t = 2.0
    x_out = comb(s, x, t)
    
    # Check results
    print(f"  k: {x_out['k']:.4f}")
    print(f"  M: {x_out['M']:.2f} g/mol")
    print(f"  T: {x_out['T']:.2f} K")
    print(f"  R: {x_out['R']:.2f} J/kg-K")
    print(f"  rho: {x_out['rho']:.4f} kg/m³")
    print(f"  cstar: {x_out['cstar']:.2f} m/s")
    
    # Test after burn time
    t = 6.0
    k_before = x_out['k']
    x_out = comb(s, x, t)
    print(f"  k after tburn: {x_out['k']:.4f}")
    assert x_out['k'] == k_before, "Properties should not change after tburn"
    
    print("  comb test passed!")


def run_all_tests():
    """Run all test functions"""
    print("Running hres physics module tests...\n")
    
    test_chamber()
    print("")
    
    test_nozzle()
    print("")
    
    test_shift_of()
    print("")
    
    test_comb()
    print("")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
