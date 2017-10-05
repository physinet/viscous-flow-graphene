import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.patches import Rectangle

from contact import Contact
import constants

class FieldSimulation():
    def __init__(self, jx, jy, L, W):
        '''
        Calculate fields from a given 2D current distribution using discretized
        Biot-Savart law.

        Parameters
        ----------
        jx, jy: matrices of current density components (A/m)
        L, W: dimensions of grid in x and y directions (um)
        '''
        self.jx, self.jy = jx, jy
        self.L = L*1e-6
        self.W = W*1e-6

        self.N, self.M = self.jx.shape
        self.dx, self.dy = self.L/self.M, self.W/self.N
        x = np.linspace(0, self.L, self.M)
        y = np.linspace(0, self.W, self.N)
        self.X, self.Y = np.meshgrid(x, y)


    def calc_Bz(self, z0=1e-6):
        '''
        z0: height above sample
        '''
        N, M, X, Y, dx, dy = self.N, self.M, self.X, self.Y, self.dx, self.dy
        jx, jy = self.jx, self.jy

        Bz = np.full(jx.shape, np.nan)
        for m in range(M):
            for n in range(N):
                num = jx * (Y[n,m] - Y) - jy * (X[n,m] - X)
                denom = ((X[n,m]-X)**2 + (Y[n,m]-Y)**2 + z0**2)**(3/2)
                dB = num/denom
                Bz[n,m] = dB[~np.isnan(dB)].sum()
        self.Bz = Bz * constants.mu0 / (4*np.pi) * dx * dy
