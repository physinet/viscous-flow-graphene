import numpy as np
import matplotlib.pyplot as plt

class Contact():
    def __init__(self, side, center, Npts, I, kind='float'):
        '''
        Parameters
        ----------
        side: (string) Side of device contact is on. ('b', 't', 'l', or 'r')
        center: (float) position (um) of center of contact
        Npts: (float) # grid points comprising the contact.
            Recommendeded to do odd numbers for symmetry.
        I: (float) current bias (Ampere)
        kind: (string) ('source', 'drain', or 'float')
        '''

        assert side in ('b','t', 'l','r') # bottom, top, left, right
        assert kind in ('source', 'drain', 'float')

        self.side = side
        self.center = center * 1e-6
        self.Npts = Npts

        if kind == 'float':
            self.current = 0
        else:  # source or drain
            if side in ('t', 'r'):
                self.current = -I  # e.g., current sourced from top is in -y direction
            else:
                self.current = I
            if kind == 'drain':
                self.current *= -1  # current should go the other way
        self.kind = kind

    def generate_coords(self, x, y):
        '''
        x, y: array of position coordinates

        '''
        self.coords = []

        # Actual width quantized to grid
        if self.side in ('t','b'):
            if self.Npts > len(x):
                self.Npts = len(x)
            self.width = self.Npts * (x[1]-x[0])
        else:
            if self.Npts > len(y):
                self.Npts = len(y)
            self.width = self.Npts * (y[1]-y[0])
        print('Contact', self.kind, 'actual width', self.width*1e6, 'um')

        # Center point shifted to nearest grid point
        if self.side in ('t','b'):
            self.center = x[abs(x-self.center).argmin()]
        else:
            self.center = y[abs(y-self.center).argmin()]
        print('Center shifted to %.4f' %(self.center*1e6))

        if self.side == 'b':
            where = np.where((abs(x - self.center) <= self.width/2*1.001))[0] # a bit extra to help roundoff error
            for i, idx in enumerate(where):
                self.coords.append((idx, 0))

        if self.side == 't':
            where = np.where((abs(x - self.center) <= self.width/2*1.001))[0]

            for i, idx in enumerate(where):
                self.coords.append((idx, len(y)-1))

        if self.side == 'l':
            where = np.where((abs(y - self.center) <= self.width/2*1.001))[0]

            for i, idx in enumerate(where):
                self.coords.append((0, idx))

        if self.side == 'r':
            where = np.where((abs(y - self.center) <= self.width/2*1.001))[0]

            for i, idx in enumerate(where):
                self.coords.append((len(x)-1, idx))

        # print(abs(x-self.center), self.width/2)

        if len(self.coords) != self.Npts:
            raise Exception('Not enough coordinates. Something went wrong.')
