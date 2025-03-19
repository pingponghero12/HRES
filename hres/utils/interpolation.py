import numpy as np
from scipy import interpolate

def interp1x(x, y, xi):
    """
    1-dimensional linear interpolation for point xi given arrays X and Y
    Replacement for MATLAB's interp1x function
    
    Parameters:
    -----------
    x : array_like
        Array of x values
    y : array_like
        Array of y values
    xi : float
        Point at which to interpolate
        
    Returns:
    --------
    float
        Interpolated value at xi
    """
    # Handle edge cases
    if xi <= x[0]:
        return y[0]
    elif xi >= x[-1]:
        return y[-1]
    else:
        # Use numpy's interp which is optimized for this purpose
        return np.interp(xi, x, y)


def interp2x(x, y, z, xi, yi):
    """
    2-dimensional linear interpolation for point (xi, yi) given arrays X, Y, and Z
    Replacement for MATLAB's interp2x function
    
    Parameters:
    -----------
    x : array_like
        1D array of x values (columns) in ascending order
    y : array_like
        1D array of y values (rows) in ascending order
    z : array_like
        2D array of z values with shape (len(y), len(x))
    xi : float
        x-value at which to interpolate
    yi : float
        y-value at which to interpolate
        
    Returns:
    --------
    float
        Interpolated value at (xi, yi)
    """
    # Handle edge cases for x
    if xi <= x[0]:
        xi = x[0]
    elif xi >= x[-1]:
        xi = x[-1]
    
    # Handle edge cases for y
    if yi <= y[0]:
        yi = y[0]
    elif yi >= y[-1]:
        yi = y[-1]
    # Convert the grid to a set of points for LinearNDInterpolator
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.vstack((xgrid.flatten(), ygrid.flatten())).T
    values = z.flatten()
    
    # Create the interpolator with the points and values
    interp_func = interpolate.LinearNDInterpolator(points, values)
    
    # Query the interpolator at point (xi, yi)
    result = interp_func(np.array([xi, yi]))
    
    return float(result[0])
