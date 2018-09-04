'''
When simulating current density and the fields generated by such a current
density distribution in a device, it is useful to be able to "tack on" the
fields generated by the leads afterwards (assuming the simulation only included
contacts at the edges that violate the continuity equation).

These can be included in the current density before the field is calculated;
however, it is useful to treat them as semi-infinite, originating at an infinity
and terminating at the edge of the device. The following functions calculate the
field originating from such a wire with finite width.
'''
import numpy as np
from . import constants

def Blead(x, y, x0, y0, z0, w, I, direction, infinity):
    '''
    x/y (array-like): 1D or 2D array of x/y values
    x0/y0: termination point on device (m)
    z0: height above device (m)
    w: width of wire (m)
    I: bias current (A). Positive current flows in +`direction`
    direction ('x' or 'y'): direction of wire
    infinity ('+' or '-'): which infinity the wire extends to
    '''
    assert direction in ('x', 'y')
    assert infinity in ('+', '-')

    if direction == 'x':  # calculation assumes vertical
        x, y, x0, y0 = y, x, y0, x0  # swap coordinates
        I *= -1  # fix sign

    if infinity == '+':  # calculation assumes -infinity
        alpha = -1  # will flip the sign of the last two terms
    else:
        alpha = 1

    # Helper variables
    dxn = x-x0-w/2
    dxp = x-x0+w/2
    dy = y-y0

    num = dxn**2 + z0**2
    denom = dxp**2 + z0**2
    term1 = np.log(num/denom)

    num = dy - np.sqrt(dxn**2 + dy**2 + z0**2)
    denom = dy - np.sqrt(dxp**2 + dy**2 + z0**2)
    term2 = -np.log(num/denom)

    num = dy + np.sqrt(dxn**2 + dy**2 + z0**2)
    denom = dy + np.sqrt(dxp**2 + dy**2 + z0**2)
    term3 = np.log(num/denom)

    return constants.mu0 * I / (8 * np.pi * w) * (term1 + alpha*(term2 + term3))
