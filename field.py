import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.patches import Rectangle
from scipy.interpolate import interp2d

from .contact import Contact
from . import constants

class FieldSimulation():
    def __init__(self, x, y, jx, jy, x0, y0, z0=1e-7):
        '''
        Calculate fields from a given 2D current distrubiton using descretized
        Biot-Savart law.

        Parameters
        ----------
        x, y: matrices (meshgrid) of points in space at which to calculate field
           (m)
        jx, jy: matrices of current density components (A/m)
        x0, y0: meshgrid of points (m) over which jx, jy are defined.
        z0: height above sample (m). Default: 100 nm
        '''
        self.x, self.y = x, y
        self.jx, self.jy = jx, jy
        self.x0, self.y0, self.z0 = x0, y0, z0

        self.N, self.M = self.x.shape
        self.dx0, self.dy0 = np.diff(self.x0[0,:])[0], np.diff(self.y0[:,0])[0]

    def calc_Bz(self):
        jx, jy = self.jx, self.jy
        x, y = self.x, self.y
        x0, y0, z0 = self.x0, self.y0, self.z0

        Bz = np.full(x.shape, np.nan)
        # Loop over array of points where field desired
        # Will numerically integrate over the grid of jx, jy defined at x0, y0
        for m in range(self.M):
            for n in range(self.N):
                num = jx * (y[n,m] - y0) - jy * (x[n,m] - x0)
                denom = ((x[n,m]-x0)**2 + (y[n,m]-y0)**2 + z0**2)**(3/2)
                dB = num/denom
                Bz[n,m] = dB[~np.isnan(dB)].sum()
        self.Bz = Bz * constants.mu0 / (4*np.pi) * self.dx0 * self.dy0

    def interpolate(self):
        '''
        Interpolate results to a new grid spacing.
        Createes a function fBz that accepts points (x,y).
        '''
        f = interp2d(self.x, self.y, self.Bz)

        self.fBz = f
