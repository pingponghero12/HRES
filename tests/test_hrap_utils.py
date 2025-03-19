import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from pathlib import Path

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from hres.utils.interpolation import interp1x, interp2x
from hres.utils.nox import nox
from hres.utils.impulse import impulse
from hres.utils.const_of import const_of

def test_interp1x():
    print("Testing interp1x...")
    
    # Test data
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 4, 9, 16, 25])
    
    # Test interpolation at various points
    test_points = [-1, 0, 0.5, 2.5, 5, 6]
    expected = [0, 0, 0.5, 6.5, 25, 25]
    
    for i, xi in enumerate(test_points):
        yi = interp1x(x, y, xi)
        print(f"  interp1x({xi}) = {yi}, expected: {expected[i]}")
        assert abs(yi - expected[i]) < 1e-10, f"Error at x={xi}"
    
    print("  interp1x test passed!")


def test_interp2x():
    print("Testing interp2x...")
    
    # Test data - a simple 3x4 grid
    x = np.array([1, 2, 3, 4])  # columns
    y = np.array([10, 20, 30])  # rows
    z = np.array([
        [100, 200, 300, 400],   # z values for y=10
        [110, 210, 310, 410],   # z values for y=20
        [120, 220, 320, 420]    # z values for y=30
    ])
    
    # Test points and expected values
    test_points = [
        (1, 10, 100),    # corner point
        (4, 30, 420),    # corner point
        (2.5, 20, 260),  # interpolated point
        (0, 10, 100),    # out of range x (low)
        (5, 10, 400),    # out of range x (high)
        (2, 0, 200),     # out of range y (low)
        (2, 40, 220)     # out of range y (high)
    ]
    
    for xi, yi, expected in test_points:
        zi = interp2x(x, y, z, xi, yi)
        print(f"  interp2x({xi}, {yi}) = {zi}, expected: {expected}")
        assert abs(zi - expected) < 1e-10, f"Error at point ({xi}, {yi})"
    
    print("  interp2x test passed!")


def test_nox():
    print("Testing nox...")
    
    # Test at two different temperatures
    test_temps = [273.15, 293.15]  # 0°C and 20°C
    
    for T in test_temps:
        props = nox(T)
        print(f"  NOx properties at {T}K:")
        for key, value in props.items():
            print(f"    {key}: {value}")
        
        # Basic validation
        assert props['Pv'] > 0, "Vapor pressure should be positive"
        assert props['rho_l'] > props['rho_v'], "Liquid density should be higher than vapor density"
    
    print("  nox test passed!")


def test_impulse():
    print("Testing impulse...")
    
    # Test cases
    test_cases = [
        (1.0, 'A', 80.0),          # A class
        (2.5, 'B', 41.67),         # B class
        (100, 'G', 25.0),          # G class
        (1000, 'J', 56.25),        # J class
        (12000, 'N', 48.83)        # N class
    ]
    
    for imp, expected_class, expected_percent in test_cases:
        motor_class, percent = impulse(imp)
        print(f"  impulse({imp}) = {motor_class}-{percent:.1f}%, expected: {expected_class}-{expected_percent:.1f}%")
        assert motor_class == expected_class, f"Wrong motor class for impulse {imp}"
    
    print("  impulse test passed!")


def test_const_of():
    print("Testing const_of...")
    
    # Create test simulation parameters
    s = {
        'dt': 0.01,                # time step in seconds
        'const_OF': 5.0,           # constant O/F ratio
        'prop_Rho': 1050.0,        # propellant density in kg/m³
        'grn_L': 0.5               # grain length in meters
    }
    
    # Create test state
    x = {
        'mdot_o': 0.1,             # oxidizer mass flow rate in kg/s
        'grn_ID': 0.05,            # grain inner diameter in meters
        'm_f': 1.0                 # fuel mass in kg
    }
    
    # Call the function
    x = const_of(s, x)
    
    # Expected values
    expected_mdot_f = 0.02         # mdot_o / const_OF = 0.1 / 5.0 = 0.02
    expected_rdot = 0.02 / (s['prop_Rho'] * np.pi * 0.05 * s['grn_L'])
    expected_grn_ID = 0.05 + 2 * expected_rdot * s['dt']
    expected_m_f = 1.0 - expected_mdot_f * s['dt']
    
    # Check results
    print(f"  mdot_f: {x['mdot_f']}, expected: {expected_mdot_f}")
    print(f"  rdot: {x['rdot']}, expected: {expected_rdot}")
    print(f"  grn_ID: {x['grn_ID']}, expected: {expected_grn_ID}")
    print(f"  m_f: {x['m_f']}, expected: {expected_m_f}")
    
    assert abs(x['mdot_f'] - expected_mdot_f) < 1e-10, "Wrong fuel mass flow rate"
    assert abs(x['rdot'] - expected_rdot) < 1e-10, "Wrong regression rate"
    assert abs(x['grn_ID'] - expected_grn_ID) < 1e-10, "Wrong grain ID"
    assert abs(x['m_f'] - expected_m_f) < 1e-10, "Wrong fuel mass"
    
    print("  const_of test passed!")


def run_all_tests():
    """Run all test functions"""
    print("Running hres utility tests...\n")
    
    test_interp1x()
    print("")
    
    test_interp2x()
    print("")
    
    test_nox()
    print("")
    
    test_impulse()
    print("")
    
    test_const_of()
    print("")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
