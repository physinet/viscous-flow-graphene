import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle

from contact import Contact
from simulation import Simulation
import constants

class ViscousSimulation(Simulation):
    def __init__(self, L, W, M, N, I, n, mu, nu, l, contacts, bc):
        '''
        Parameters
        ----------
        L: length of device (x direction) (um)
        W: width of device (y direction) (um)
        M: number of grid points in the x direction
        N: number of grid points in the y direction
        I: magnitude of current bias (A)
        n: carrier density (cm^-2). Typical value: 1e12
        mu: carrier mobility (cm^2/V*s). Typical value: 1e5
        nu: kinematic viscosity (m^2/s). Typical value: 0.1
        l: mean free path for scattering (um). Typical value: 1
        contacts: list of Contact objects
        bc: either 'no-slip' or 'free'
        '''
        self.L = L*1e-6
        self.W = W*1e-6
        self.M = M
        self.N = N
        self.I = I

        self.dx = self.L/self.M
        self.dy = self.W/self.N

        # Position coordinates of the centers of the cells
        self.x = np.linspace(-self.L/2, self.L/2, M, endpoint=False) + self.dx/2
        self.y = np.linspace(0, self.W, N, endpoint=False) + self.dy/2

        self.n = n * 100**2  # to m^-2
        self.mu = mu / 100**2  # to m^2/V*s
        self.nu = nu
        self.l = l*1e-6  # to m

        self.D = np.sqrt(nu * self.l / constants.vF)  # units of length
        self.sigma = self.n * constants.e * self.mu  # Conductivity (S)
        self.rho = 1 / self.sigma  # resistivity, Ohm/sq

        self.contacts = contacts

        self.setup_contacts()
        self.setup_indexing()

        self.bc = bc


    def add_contact_boundary_equations(self):
        i = self.i
        idx_phi = self.idx_phi
        idx_u = self.idx_u
        idx_v = self.idx_v

        for contact in self.contacts:
            # First do the "equal potentials" equations
            for j, pt in enumerate(contact.coords):
                x, y = pt
                if j == len(contact.coords)-1:  # Skip the last one
                    pass
                else:  # do this equation on all other points within the contact
                    self.A[i, idx_phi(x, y)] = 1
                    if contact.side in ('t', 'b'):
                        # if contact is on top/bottom, we want to subtract the
                        # potential one grid point to the right
                        self.A[i, idx_phi(x+1, y)] = -1
                    else:
                        self.A[i, idx_phi(x, y+1)] = -1
                    i = i+1

            # Now do the "sum all currents" equation
            for j, pt in enumerate(contact.coords):
                x, y = pt
                if contact.side == 't':
                    self.A[i, idx_v(x+1, y+1)] = 1
                if contact.side == 'b':
                    self.A[i, idx_v(x+1, y)] = 1
                if contact.side == 'r':
                    self.A[i, idx_u(x+1, y+1)] = 1
                if contact.side == 'l':
                    self.A[i, idx_u(x, y+1)] = 1

            if contact.side in ('t','b'):
                self.b[i] = contact.current / self.dx
            else:
                self.b[i] = contact.current / self.dy

            i = i+1
        self.i = i


    def add_parallel_boundary_equations(self):
        '''
        Add boundary conditions for components of current parallel to edges.
        For no-slip boundary conditions, tangential velocities are antiparallel
            just inside and just outside a surface.
        For free surface boundary conditions, tangential velocities are parallel
            just inside and outside a surface.
        '''
        # multiplier to set boundary conditions
        if self.bc == 'no-slip':
            a = 1
        elif self.bc == 'free':
            a = -1

        i = self.i
        M = self.M
        N = self.N
        idx_phi = self.idx_phi
        idx_u = self.idx_u
        idx_v = self.idx_v

        # Parallel boundary conditions
        for m in range(1, M): # skip corners
            for n in [0, N]:  # bottom and top edges
                self.A[i, idx_u(m, n)] = 1
                self.A[i, idx_u(m, n + 1)] = a
                i = i+1

        for n in range(1, N):
            for m in [0, M]:  # left and right edges
                self.A[i, idx_v(m, n)] = 1
                self.A[i, idx_v(m + 1, n)] = a
                i = i+1
        self.i = i


    def add_perp_boundary_equations(self):
        '''
        Add equations prescribing that all currents perpendicular to the sample
        edge are zero.
        '''
        i = self.i
        M = self.M
        N = self.N
        idx_phi = self.idx_phi
        idx_u = self.idx_u
        idx_v = self.idx_v
        for m in range(M):  # loop over potential grid points in the x direction
            # bottom edge of potential grid
            if self.is_contact(m, 0) in ('t', 'b'):
                pass # pass, don't continue! Or else we skip the top edge!!
            else:
                self.A[i, idx_v(m+1, 0)] = 1  # current labels are offset from grid points
                i = i+1

            # top edge of potential grid
            if self.is_contact(m, N-1) in ('t', 'b'):
                pass
            else:
                self.A[i, idx_v(m+1, N)] = 1  # current labels are offset from grid points
                i = i+1

        for n in range(N):  # loop over potential grid points in the y direction
            # left edge
            if self.is_contact(0, n) in ('l', 'r'):
                pass
            else:
                self.A[i, idx_u(0, n+1)] = 1  # current labels are offset from grid points
                i = i+1

            # right edge
            if self.is_contact(M-1, n) in ('l', 'r'):
                pass
            else:
                self.A[i, idx_u(M, n+1)] = 1  # current labels are offset from grid points
                i = i+1
        self.i = i


    def add_continuity_equations(self):
        '''
        Add continuity equations to the coefficient matrix A. We write down this
        equation for each cell, and there are MxN cells.
        '''
        i = self.i
        M = self.M
        N = self.N
        idx_phi = self.idx_phi
        idx_u = self.idx_u
        idx_v = self.idx_v
        dx = self.dx
        dy = self.dy

        for m in range(M):
            for n in range(N):
                # multiply through by dx
                self.A[i, idx_u(m + 1, n + 1)] = 1
                self.A[i, idx_u(m, n + 1)] = -1
                self.A[i, idx_v(m + 1, n + 1)] = dx / dy
                self.A[i, idx_v(m + 1, n)] = -dx / dy

                i += 1
        self.i = i


    def add_navier_stokes_equations(self):
        '''
        Add Navier-Stokes equations to the coefficient matrix A. We only write
        this equation for the internal bonds, of which there are (M-1)xN
        in the x direction and Mx(N-1) in the y direction.
        '''
        dx = self.dx
        dy = self.dy
        D = self.D
        i = self.i
        M = self.M
        N = self.N
        idx_phi = self.idx_phi
        idx_u = self.idx_u
        idx_v = self.idx_v

        for m in range(M - 1):
            for n in range(N):
                self.A[i, idx_phi(m + 1, n)] = 1 / dx * 1e-6
                self.A[i, idx_phi(m, n)] = -1 / dx * 1e-6

                self.A[i, idx_u(m, n + 1)] = D**2 / dx**2
                self.A[i, idx_u(m + 2, n + 1)] = D**2 / dx**2
                self.A[i, idx_u(m + 1, n + 1)] = -2*D**2*(1/dx**2 + 1/dy**2) - 1
                self.A[i, idx_u(m + 1, n)] = D**2 / dy ** 2
                self.A[i, idx_u(m + 1, n + 2)] = D**2 / dy ** 2

                i += 1

        for m in range(M):
            for n in range(N - 1):

                self.A[i, idx_phi(m, n + 1)] = 1 / dy *1e-6
                self.A[i, idx_phi(m, n)] = -1 / dy * 1e-6

                self.A[i, idx_v(m, n + 1)] = D**2 / dx**2
                self.A[i, idx_v(m + 2, n + 1)] = D**2 / dx**2
                self.A[i, idx_v(m + 1, n + 1)] = -2*D**2*(1/dx**2 + 1/dy**2) - 1
                self.A[i, idx_v(m + 1, n)] = D**2 / dy ** 2
                self.A[i, idx_v(m + 1, n + 2)] = D**2 / dy ** 2

                i += 1
        self.i = i



    def add_potential_equation(self):
        '''
        Fix potential at one contact to zero. Otherwise, there is no unique
        solution, because the problem only involves differences in potentials.

        Turns out there is a solution even if this equation isn't added.
        '''
        # add a row
        self.A = sp.vstack((self.A, np.zeros(self.A.shape[0])), format='dok')
        self.A[-1, self.idx_phi(0, 0)] = 1  # add the equation

        self.b = np.append(self.b, [0])  # specify the potential = 0


    def check_continuity(self):
        '''
        Generates a matrix of continuity equations evaluated in each cell.
        If the currents follow the continuity equation, all elements should be near zero.
        Must do this without deleting phantoms.
        '''
        M = self.M
        N = self.N
        u, v, phi = self.u, self.v, self.phi

        A = np.full((N,M), np.nan)

        for m in range(M):
            for n in range(N):
                a = u[n+1, m+1] - u[n+1, m]
                a += (v[n+1, m+1] - v[n, m+1])*self.dx/self.dy
                A[n, m] = a

        return A


    def check_navier_stokes(self):
        '''
        Generates two matrices (x,y) of Navier-Stokes equations evaluated on the bonds.
        If the currents and potentials obey the equations, should get all zeros.
        Must do this without deleting phantoms.
        '''
        M = self.M
        N = self.N
        u, v, phi = self.u, self.v, self.phi
        D = self.D

        Ax = np.full((N,M-1), np.nan)
        Ay = np.full((N-1,M), np.nan)

        for m in range(M - 1):
            for n in range(N):
                a = (phi[n,m+1] - phi[n,m])/self.dx * self.n * -constants.e * self.mu
                a += D**2 * (u[n+1,m] + u[n+1, m+2] - 2*u[n+1,m+1])/self.dx**2
                a += D**2 * (u[n,m+1] + u[n+2, m+1] - 2*u[n+1,m+1])/self.dy**2
                a -= u[n+1, m+1]

                Ax[n,m] = a

        for m in range(M):
            for n in range(N - 1):
                a = (phi[n+1,m] - phi[n,m])/self.dy * self.n * -constants.e * self.mu
                a += D**2 * (v[n+1,m] + v[n+1, m+2] - 2*v[n+1,m+1])/self.dx**2
                a += D**2 * (v[n,m+1] + v[n+2, m+1] - 2*v[n+1,m+1])/self.dy**2
                a -= v[n+1, m+1]

                Ay[n,m] = a

        return Ax, Ay


    def draw_contacts(self, ax, offset=(0,0)):
        for c in self.contacts:
            if c.side == 't':
                width = len(c.coords)
                height = 1
            elif c.side == 'b':
                width = len(c.coords)
                height = -1
            elif c.side == 'r':
                width = 1
                height = len(c.coords)
            elif c.side == 'l':
                width = -1
                height = len(c.coords)

            colors = {'drain': 'k', 'source': 'b', 'float': 'w'}

            pos = (c.coords[0][0] + offset[0] - .5, c.coords[0][1] + offset[1])
            r = Rectangle(pos, width, height, zorder=6, fc=colors[c.kind], ec='k', alpha=.3)
            r.set_clip_on(False) # make it show up outside the axes
            ax.add_patch(r)


    def draw_grid(self):
        '''
        Draw grid showing cell structure, potential grid points, velocity grid
        points, and contact grid points.
        '''
        M, N = self.M, self.N
        fig, ax = plt.subplots()
        for n in range(N):
            ax.axhline(n)
            for m in range(M):
                ax.axvline(m)
                color = '.k'
                for c in self.contacts:
                    if (m, n) in c.coords:
                        color = '.r'
                ax.plot(m+.5, n+.5, color)

        for n in range(N+2):
            for m in range(M+1):
                ax.plot([m-.125, m+.125], [n-.5, n-.5], color='C1')  # u grid
        for n in range(N+1):
            for m in range(M+2):
                ax.plot([m-.5, m-.5], [n-.125, n+.125], color='g')  # v grid

        ax.axhline(M)
        ax.axvline(N)
        ax.set_xlim(-1, M+1)
        ax.set_ylim(-1, N+1)


    def extract_grids(self, delete_phantoms=True):
        '''
        Extract grids of potentials and currents from the solution vector x.
        Original current components stored as u and v, interpolated current
        components matching the grid of potentials stored as jx and jy.

        Parameters
        ----------
        delete_phantoms: (bool) If True, will remove phantom currents.
        '''
        X = self.X.copy()
        Nphi = self.Nphi
        N = self.N
        M = self.M
        Nu = self.Nu

        self.phi = X[:Nphi].reshape(self.idx_phi_mat.shape)

        us = X[Nphi : Nphi + Nu - 4]
        j=0
        self.u = np.full((N + 2, M+1), np.nan)  # shape (y, x)
        for n in range(N+2):
            for m in range(M+1):
                if m in (0, M):
                    if n in (0, N+1):
                        continue
                self.u[n,m] = us[j]
                j = j+1

        vs = X[Nphi + Nu - 4 :]
        j=0
        self.v = np.full((N+1, M + 2), np.nan)  # shape (y, x)
        for n in range(N + 1):
            for m in range(M + 2):
                if m in (0, M+1):
                    if n in (0, N):
                        continue
                self.v[n,m] = vs[j]
                j = j+1

        # Delete phantom variables
        if delete_phantoms:
            self.u = self.u[1:-1, :]
            self.v = self.v[:, 1:-1]

        # Create version of the current grid with the same grid positions as the potential
        self.jx = np.full(self.phi.shape, np.nan)
        self.jy = np.full(self.phi.shape, np.nan)

        for m in range(M):
            for n in range(N):
                self.jx[n,m] = (self.u[n, m] + self.u[n, m + 1]) / 2
                self.jy[n,m] = (self.v[n, m] + self.v[n+1, m]) / 2


        self.phi -= self.phi.mean()  # Center zero of potential for symmetry
        self.phi /= -(self.n * self.mu * constants.e) * 1e6  # Scale Phi to Voltage # we scaled out by 1e6 in the Navier-Stokes equations.


    def interpolate(self, newM, newN):
        '''
        Interpolate results to a new grid spacing.
        Interpolated matrices stored as phi1, jx1, jy1.
        Interpolated position variables stored as x1, y1.
        '''
        self.x1 = np.linspace(self.x.min(), self.x.max(), newM)
        self.y1 = np.linspace(self.y.min(), self.y.max(), newN)

        for var in ['phi', 'jx', 'jy', 'omega']:
            if hasattr(self, var):
                f = interp2d(self.x, self.y, getattr(self, var))

                setattr(self, (var+'1'), f(self.x1, self.y1))



    def is_contact(self, m, n):
        '''
        Parameters
        ----------
        m: x grid point
        n: y grid point

        Returns
        ----------
        If the given grid point lies on a contact, then return the side of that
        contact. Else, return False.
        '''
        for c in self.contacts:
            if (m, n) in c.coords: # flipped 9/21 3:48pm. Bug!
                return c.side
        return False


    def plot_A(self):
        '''
        Plot the coefficient matrix as a heatmap. Not practical for large matrix
        '''
        A = self.A.toarray()

        A = A/abs(A)

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(A, interpolation='none', cmap='coolwarm',
                    vmin = -1, vmax = 1
                 )

        # Add the grid
        import matplotlib.ticker as plticker
        if self.N<5:
            myInterval=1
            loc = plticker.MultipleLocator(base=myInterval)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)
            ax.grid(which='major', axis='both', linestyle='-')


    def plot_results(self):
        '''
        Generates three colorplots for the potential, x current, and y current.
        '''
        phi = self.phi
        u = self.u
        v = self.v
        N = self.N
        M = self.M
        L = self.L
        W = self.W

        phi2 = phi/np.nanmax(abs(phi))
        u2 = u/np.nanmax(abs(u))
        v2 = v/np.nanmax(abs(v))

        cm = 'coolwarm'
        size = max(L/W, W/L)
        fig, ax = plt.subplots(ncols=3,figsize=(18*size, 9*size))
        im0 = ax[0].imshow(phi2, cmap='viridis', origin='lower', vmin=-np.nanmax(abs(phi2)), vmax=np.nanmax(abs(phi2)), aspect=W/L)
        im1 = ax[1].imshow(u2, cmap=cm, origin='lower', vmin=-np.nanmax(abs(u2)), vmax=np.nanmax(abs(u2)), aspect=W/L)
        im2 = ax[2].imshow(v2, cmap=cm, origin='lower', vmin=-np.nanmax(abs(v2)), vmax=np.nanmax(abs(v2)), aspect=W/L)

        # Label cells with values
        if M < 14 and N < 14:
            for k, var in enumerate([phi2, u2, v2]):
                for (j, i), label in np.ndenumerate(var):
                    label = '%.2f' %label
                    ax[k].text(i, j, label, ha='center', va='center')
        self.draw_contacts(ax[0], offset=(0,0))
        self.draw_contacts(ax[1], offset=(0.5,0))
        self.draw_contacts(ax[2], offset=(0,0.5))


    def plot_streamplot(self, density=1.1):
        '''
        Generates a streamplot of the currents with the potential as the color
        background.
        '''
        phi = self.phi
        jx = self.jx
        jy = self.jy
        M = self.M
        N = self.N
        L = self.L
        W = self.W

        c = np.sqrt(jx**2+jy**2)
        c /= c.max()

        size = max(L/W, W/L)
        fig, ax = plt.subplots(figsize=(5*size, 5*size))
        ax.imshow(phi, cmap='viridis', origin='lower', vmin=-np.nanmax(abs(phi)), vmax=np.nanmax(abs(phi)), aspect=W/L)
        ax.streamplot(np.array(range(M)), np.array(range(N)), jx, jy, density=density, color=c, cmap='Greys')

        self.draw_contacts(ax)


    def run(self, delete_phantoms=True, extra_equation=True):
        '''
        Run the simulation.

        Parameters
        ----------
        delete_phantoms (bool): If True, delete phantom currents.
        extra_equation (bool): If True, add an extra equation specifying
            one of the potentials.
        '''
        self.i = 0  # index to keep track of each equation added
        self.setup_matrices()

        self.add_continuity_equations()
        self.add_navier_stokes_equations()
        self.add_perp_boundary_equations()
        self.add_contact_boundary_equations()
        self.add_parallel_boundary_equations(bc)
        if extra_equation:
            self.add_potential_equation()

        self.solve()
        self.extract_grids(delete_phantoms)

        self.save()


    def setup_contacts(self):
        for c in self.contacts:
            c.generate_coords(self.x, self.y)


    def setup_indexing(self):
        '''
        Setup helper functions that label all the potentials and currents with
        distinct indices. Creates lookup functions to convert from grid points
        to indices.
        '''
        N = self.N
        M = self.M

        self.idx_phi_mat = np.reshape(range(M*N), [N, M])  # shape is y, x
        Nphi = len(self.idx_phi_mat.flatten())

        i=0
        self.idx_u_mat = np.full((N+2, M+1), np.nan)  # shape (y, x)
        for n in range(N+2):
            for m in range(M+1):
                if m in (0, M):
                    if n in (0, N+1):
                        continue
                self.idx_u_mat[n,m] = i + Nphi
                i = i+1
        Nu = len(self.idx_u_mat.flatten())


        i=0
        self.idx_v_mat = np.full((N+1, M+2), np.nan)
        for n in range(N+1):
            for m in range(M+2):
                if m in (0, M+1):
                    if n in (0, N):
                        continue
                self.idx_v_mat[n,m] = i + Nphi + Nu - 4
                i = i+1
        Nv = len(self.idx_v_mat.flatten())

        def idx_phi(m,n):
            assert m >= 0 # make sure we don't accidentally pass -1 and roll to the end of the matrix.
            assert n >= 0
            return self.idx_phi_mat[n,m]
        self.idx_phi = idx_phi

        def idx_u(m,n):
            assert m >= 0
            assert n >= 0
            return int(self.idx_u_mat[n,m])
        self.idx_u = idx_u

        def idx_v(m,n):
            assert m >= 0
            assert n >= 0
            return int(self.idx_v_mat[n,m])
        self.idx_v = idx_v


        self.Nphi = Nphi
        self.Nu = Nu
        self.Nv = Nv


    def setup_matrices(self):
        '''
        Set up empty coefficient and consant matrices.
        We're preparing to solve Ax = b, where x is a vector of all the unknown
        potentials and currents.
        '''
        s = self.Nphi + self.Nu + self.Nv - 8
        self.A = sp.dok_matrix((s,s))
        self.b = np.zeros(s)


    def solve(self, normalize=True, alg='lsmr', tol=1e-11):
        '''
        Solves for all unknown variables using least squares.
        Normalize (bool): Should we normalize by max of each row?
        alg: spsolve, lsqr or lsmr - all scipy.sparse methods
        tol: tolerance atol (and btol) for lsmr.
        '''
        t = time.time()

        # Normalize all rows of A by the max value in each row
        if normalize:
            from sklearn.preprocessing import normalize

            b = sp.dok_matrix((self.b,)).T
            C = sp.hstack((self.A, b))  # augmented matrix
            Cn = normalize(C, norm='max', axis=1)  # normalize s.t. max(row) = 1
            An = Cn[:, :-1]  # split off A again
            bn = Cn[:, -1].toarray()  # b is the last column  and cannot be sparse
        else:
            An, bn = self.A, self.b

        if alg == 'lsqr':
            returns = spla.lsqr(An, bn, atol=tol, btol=tol) # necessary to increase tolerance. lsqr also works with 1e-11 tolerance (2x slower)
        elif alg == 'lsmr':
            returns = spla.lsmr(An, bn, maxiter=1e10, atol=tol, btol=tol) # necessary to increase tolerance and use many iterations!. lsqr also works with 1e-11 tolerance (2x slower)
            if returns[1] not in (0,1,2):
                raise Exception(returns)
        elif alg == 'spsolve':
            returns = spla.spsolve(An, bn)
        self.X = returns[0]

        print('Calculation took %i seconds' %(time.time()-t))


    def vorticity(self):
        '''
        calculate vorticity from the currents.

        omega = 1/n grad cross J.
        omega_z = 1/n * (dJy/dx - dJx/dy)
        '''

        gradyjx, gradxjx = np.gradient(self.jx)
        gradxjx /= self.dx
        gradyjx /= self.dy

        gradyjy, gradxjy = np.gradient(self.jy)
        gradxjy /= self.dx
        gradyjy /= self.dy

        self.omega = 1 / self.n * (gradxjy - gradyjx)
