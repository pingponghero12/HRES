import numpy as np
from scipy import optimize

def nox(T):
    """
    Calculates thermophysical properties of saturated nitrous oxide as a function of temperature
    
    Parameters:
    -----------
    T : float
        Temperature in Kelvin
        
    Returns:
    --------
    dict
        Dictionary containing thermophysical properties
    """
    # Critical properties
    Pc = 7251000
    Tc = 309.57
    rhoc = 452
    R = 188.91
    
    # Initialize the output dictionary
    op = {}
    
    # Vapor Pressure
    a1 = -6.71893
    a2 = 1.35966
    a3 = -1.3779
    a4 = -4.051
    op['Pv'] = Pc * np.exp((1/(T/Tc))*(a1*(1-T/Tc) + 
                a2*(1-(T/Tc))**(3/2) + a3*(1-(T/Tc))**(5/2) + a4*(1-(T/Tc))**5))
    
    # Liquid Density
    b1 = 1.72328
    b2 = -0.83950
    b3 = 0.51060
    b4 = -0.10412
    op['rho_l'] = rhoc * np.exp(b1*(1-(T/Tc))**(1/3) + 
                b2*(1-(T/Tc))**(2/3) + b3*(1-(T/Tc)) + b4*(1-(T/Tc))**(4/3))
    
    # Vapor Density
    c1 = -1.00900
    c2 = -6.28792
    c3 = 7.50332
    c4 = -7.90463
    c5 = 0.629427
    op['rho_v'] = rhoc * np.exp(c1*((Tc/T)-1)**(1/3) + 
                c2*((Tc/T)-1)**(2/3) + c3*((Tc/T)-1) + c4*((Tc/T)-1)**(4/3) + 
                c5*((Tc/T)-1)**(5/3))
    
    # Specific Enthalpy
    # Liquid Enthalpy
    d1 = -200
    d2 = 116.043
    d3 = -917.225
    d4 = 794.779
    d5 = -589.587
    
    # Vapor Enthalpy
    e1 = -200
    e2 = 440.055
    e3 = -459.701
    e4 = 434.081
    e5 = -485.338
    
    # Latent Heat of Vaporization
    op['Hv'] = (e1-d1) + (e2-d2)*(1-(T/Tc))**(1/3) + (e3-d3)*(1-(T/Tc))**(2/3) + \
              (e4-d4)*(1-(T/Tc)) + (e5-d5)*(1-(T/Tc))**(4/3)
    
    # Specific Heat Capacity of Saturated Liquid
    f1 = 2.49973
    f2 = 0.023454
    f3 = -3.80136
    f4 = 13.0945
    f5 = -14.5180
    
    op['Cp'] = f1*(1 + f2*(1-(T/Tc))**(-1) + f3*(1-(T/Tc)) + 
              f4*(1-(T/Tc))**2 + f5*(1-(T/Tc))**3)
    
    # Saturated Vapor Compressibility Factor
    op['Z'] = op['Pv'] / (op['rho_v'] * R * T)
    
    return op
