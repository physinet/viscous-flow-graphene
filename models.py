from constants import e
from scipy.integrate import quad
from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt

def fourier(x, fx, hann=True):
    '''
    Returns an array of frequencies and a Fourier-transformed function with
    proper continuous normalization.

    This is for the Fourier transform with e^(-ikx)

    Example:
    N = 1000
    L = 100
    x = np.linspace(-L, L, N)
    dx = 2*L/N

    fx = np.sin(x)
    '''
    dx = x[1]-x[0]
    L = x.max()
    k = np.fft.fftfreq(len(x)) * 2 * np.pi / dx

    N = len(x)
    hann = np.hanning(N)

    fk = np.fft.fft(fx * hann)
    fk *= dx * np.exp(-1j*k*L) / np.sqrt(2*np.pi)
    return k, fk

def inv_fourier(k, fk, hann=False):
    '''
    Returns an array of positions and an inverse Fourier-transformed function with
    proper continuous normalization.

    This is for the Fourier transform with e^(+ikx)
    '''
    dk = k[1] - k[0]
    K = k.max()
    x = np.fft.fftfreq(len(k)) * 2 * np.pi / dk

    N = len(k)
    hann = np.hanning(N)

    fx = np.fft.ifft(fk * hann) * N # Normalization
    fx *= dk * np.exp(-1j*x*K) / np.sqrt(2*np.pi)
    return x, fx

def pnv(x, rho, w):
    '''
    Calculate the nonlocal resistance measured using voltage probes
    in parallel with the source/drain contacts.

    R(x) = 2 * rho/pi * ln(|coth(pi*x/2*w)|)

    Parameters
    ----------
    x: distance between probes (um)
    rho: 2D resistivity (Ohm/sq)
    w: width of device (um)
    '''
    return 2 * rho/np.pi * np.log(abs(1/np.tanh(np.pi * x / 2 / w)))

def vdP(x, rho, w):
    '''
    Calculate the nonlocal resistance measured using voltage probes
    in parallel with the source/drain contacts.

    This is the van der Pauw formula, an approximation valid for |x| >> W/pi

    R(x) = 4 * rho/pi * exp(-pi|x|/W)

    Parameters
    ----------
    x: distance between probes (um)
    rho: 2D resistivity (Ohm/sq)
    w: width of device (um)
    '''
    return 4 * rho / np.pi * np.exp(-np.pi * abs(x) / w)


class Device():
    X = None
    Y = None

    def plot_2D(self, A, vmin=None, vmax=None, aspect=None):
        '''
        Plot a 2D array
        '''

        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(A, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect=aspect, origin='lower')
        plt.colorbar(im)
        return fig, ax

class LevitovDevice(Device):
    def __init__(self, L, W, Nx, Ny, I, n, mu, nu, tau):
        '''
        https://www.nature.com/nphys/journal/vaop/ncurrent/pdf/nphys3667.pdf

        Parameters
        ----------
        L: Length of device (um, X direction)
        W: Width of device (um, Y direction)
        Nx: number of grid points (X direction)
        Ny: number of grid points (Y direction)
        I: current bias (uA)
        n: carrier density (cm^-2)
        mu: carrier mobility (cm^2/V*s)
        nm: viscosity (m^2/s)
        tau: scattering time (s)
        '''
        self.L, self.W = (L*1e-6, W*1e-6)
        self.Nx, self.Ny = (Nx, Ny)
        self.I = I*1e-6  # A to uA
        self.n = n*1e4  # cm^-2 to m^-2
        self.mu = mu/1e4  # cm^2/V*s to m^2/V*s
        self.nu = nu
        self.tau = tau

        # Quantities used by Levitov
        m = e*self.tau/self.mu  # effective mass
        self.eta = m*self.n*self.nu  # dynamic viscosity
        sigma = self.n*e*self.mu
        self.rho = 1/sigma

        self.x = np.linspace(-self.L/2, self.L/2, Nx, endpoint=False)
        self.y = np.linspace(0, self.W, Ny, endpoint=False)

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.dx = self.L/Nx
        self.dy = self.W/Ny

    def potential_viscous(self):
        '''
        This is only for the pure viscous case.
        The potential() function is for the general Ohmic-viscous case.
        Use that one.
        '''
        w = self.W

        Nx = 10000
        Ny = 10

        Y = np.linspace(0, self.W, Ny)

        K = 2 * np.pi / self.dx
        ks = np.linspace(-K, K, Nx)
        phi = np.full((Ny, Nx), np.nan)

        def phi_k(k, y):
            a_k = k * np.tanh(k*w/2) / (k*w + np.sinh(k*w))
            return a_k * (np.sinh(k*(y-w)) + np.sinh(k*y))

        for i, y in enumerate(Y):
            x, phi_x = inv_fourier(ks, phi_k(ks, y), hann=False)
            phi[i,:] = np.fft.ifftshift(phi_x.real)

        # Now interpolate to the desired grid size
        f = interp2d(np.fft.ifftshift(x), Y, phi)
        phi_int = f(self.x, self.y)

        self.phi_v = self.I * self.rho / (2 * np.pi) * phi_int


    def potential(self):
        '''
        Calculates the general ohmic-viscous potential by inverse Fourier
        transforming Eq. (22) in the supplement.
        '''
        w = self.W

        Nx = 10000
        Ny = 100

        Y = np.linspace(0, w, Ny)

        K = 2 * np.pi / self.dx
        ks = np.linspace(-K, K, Nx)
        phi = np.full((Ny, Nx), np.nan)

        print('epsilon: ',self.rho*(e*self.n*w)**2/self.eta)

        def phi_k(k, y):
            q = np.sqrt(k**2 + self.rho*(e*self.n)**2/self.eta)

            term1 = (np.exp(q*w) - 1)*q
            term2 = (k-q) * (1 - np.exp((k+q)*w))
            term3 = (k+q) * (np.exp(q*w) - np.exp(k*w))

            aplus = term1/(term2 + term3)
            aminus = np.exp(k*w) * aplus
            return (aplus * np.exp(k*y) - aminus*np.exp(-k*y))/k

        for i, y in enumerate(Y):
            x, phi_x = inv_fourier(ks, phi_k(ks, y), hann=False)
            phi[i,:] = np.fft.ifftshift(phi_x.real)

        # Now interpolate to the desired grid size
        f = interp2d(np.fft.ifftshift(x), Y, phi)
        phi_int = f(self.x, self.y)

        self.phi = self.I * self.rho / (2 * np.pi) * phi_int


    def plot_potential(self):
        if not hasattr(self, 'phi'):
            self.potential() # calculate it
        vmax = 1e-1
        return self.plot_2D(self.phi, vmin=-vmax, vmax=vmax, aspect=(self.Nx/self.Ny)*(self.W/self.L))


    def stream(self):
        Nx, Ny = self.Nx, self.Ny
        w = self.W

        N = 10000

        ks = np.linspace(-600, 600, N)  # empirical
        psi = np.full((Ny, Nx), np.nan)

        def psi_k(k, y):
            a_k = k * np.tanh(k*w/2) / (k*w + np.sinh(k*w))
            term1 = (np.exp(k*y) + np.exp(k*(w-y))) / (np.exp(k*w) + 1)
            term2 = a_k * (y*np.sinh(k*(w-y)) + (w-y)*np.sinh(k*y))
            return 1/(2*np.pi*1j*k) * (term1 + term2)

        for i, y in enumerate(self.Y[:,0]):
            x, psi_x = inv_fourier(ks, psi_k(ks, y), hann=False)
            psi[i,:] = np.interp(self.X[0,:],
                            np.fft.ifftshift(x), np.fft.ifftshift(psi_x.real))

        self.psi = psi * self.I / (self.n * e)


class PoliniDevice(Device):
    def __init__(self, L, W, Nx, Ny, x0, I, n, mu, nu, tau, x=None, y=None):
        '''
        L: Length of device (um, X direction)
        W: Width of device (um, Y direction)
        Nx: number of grid points (X direction)
        Ny: number of grid points (Y direction)
        x0: position of contacts (um, either side of origin)
        I: Current bias (A)
        n: carrier density (cm^-2)
        mu: carrier mobility (cm^2/V*s)
        nu: kinematic viscosty (m^2/s)
        tau: scattering time (s)
        x, y: the exact x and y arrays from one of my simulations
        '''
        self.L, self.W = (L*1e-6, W*1e-6)
        self.Nx, self.Ny = (Nx, Ny)
        self.x0 = x0*1e-6
        self.I = I*1e-6  # A to uA
        self.n = n*1e4  # cm^-2 to m^-2
        self.mu = mu/1e4  # cm^2/V*s to m^2/V*s
        self.nu = nu
        self.tau = tau

        # Quantities used by Polini
        self.sigma0 = self.n*e*self.mu  # conductivity
        self.D = np.sqrt(self.nu*self.tau) # characteristic viscous diffusion length

        if x is None:
            self.x = np.linspace(-self.L/2, self.L/2, Nx, endpoint=False)
            self.y = np.linspace(-self.W/2, self.W/2, Ny, endpoint=False)
        else:
            self.x, self.y = x, y

        self.X, self.Y = np.meshgrid(self.x, self.y)
        # self.W = y.max()-y.min()

        self.dx = self.L/Nx
        self.dy = self.W/Ny

    def jxjy(self):
        '''
        Calculate Jx and Jy from vorticity and potential
        J = - sigma0 grad phi + n*D^2 grad x zhat vorticity
        '''
        if not hasattr(self, 'phi'):
            self.potential()
        if not hasattr(self, 'omega'):
            self.vorticity()

        gradyphi, gradxphi = np.gradient(self.phi)
        gradxphi /= self.dx
        gradyphi /= self.dy

        gradyomega, gradxomega = np.gradient(self.omega)
        gradxomega /= self.dx
        gradyomega /= self.dy

        self.jx = -self.sigma0 * gradxphi
        self.jy = -self.sigma0 * gradyphi

        if self.D != 0: # explicitly do this to avoid nans
            self.jx += self.n * self.D**2 * gradyomega
            self.jy -= self.n * self.D**2 * gradxomega

        return self.jx, self.jy


    def plot_potential(self):
        if not hasattr(self, 'phi'):
            self.potential() # calculate it
        vmax = self.I/self.sigma0
        fig, ax = self.plot_2D(self.phi/vmax, vmin=-1, vmax=1, aspect=(self.Nx/self.Ny)*(self.W/self.L))

        return fig, ax

    def plot_streamplot(self):
        '''
        Generates a streamplot of the currents with the potential as the color
        background.
        '''
        if not hasattr(self, 'jx'):
            self.jxjy()

        fig, ax = plt.subplots(figsize=(5*self.L/self.W, 5*self.L/self.W))
        vmax = self.I/self.sigma0
        ax.imshow(self.phi, cmap='coolwarm', origin='lower', vmin=-vmax, vmax=vmax)
        ax.streamplot(np.array(range(self.Nx)), np.array(range(self.Ny)), self.jx, self.jy, color='k')


    def plot_vorticity(self):
        if not hasattr(self, 'omega'):
            self.vorticity() # calculate it
        vmax = 2*self.I/(e*self.n*self.W**2)
        return self.plot_2D(self.omega, vmin=-vmax, vmax=vmax, aspect=(self.Nx/self.Ny)*(self.W/self.L))


    def potential(self):
        '''
        Eq. (37) from:
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.165433
        '''
        X = self.X
        Y = self.Y
        I = self.I
        sigma0 = self.sigma0
        x0 = self.x0
        D = self.D
        W = self.W
        pi, sin, cos, exp, log = np.pi, np.sin, np.cos, np.exp, np.log

        def F1(xx):
            x, y = xx/W, Y/W

            inthelog1 = (1 + exp(-2*pi*abs(x)) + 2*sin(pi*y)*exp(-pi*abs(x))) / \
                (1 + exp(-2*pi*abs(x)) - 2*sin(pi*y)*exp(-pi*abs(x)))
            term1 = 1/(4*pi) * log(inthelog1)

            inthelog2 = 1 + exp(-4*pi*abs(x)) + 2*cos(2*pi*y)*exp(-2*pi*abs(x))
            term2 = 1/(4*pi) * log(inthelog2)

            term3 = abs(x)/2

            return term1 + term2 + term3

        def F2(xx):
            x, y = xx/W, Y/W

            term1 = -pi * sin(pi*y)*exp(-pi*abs(x))*(1+exp(-2*pi*abs(x)))
            term2 = 1 + exp(-4*pi*abs(x)) - 2*(cos(2*pi*y)+2)*exp(-2*pi*abs(x))
            term3 = 2*pi*exp(-2*pi*abs(x))
            term4 = cos(2*pi*y)*(1+exp(-4*pi*abs(x))) + 2*exp(-2*pi*abs(x))
            term5 = 1 + exp(-4*pi*abs(x)) + 2*cos(2*pi*y)*exp(-2*pi*abs(x))

            return (term1*term2 - term3*term4) / term5**2

        phi = I/sigma0 * (
            F1(X+x0) - F1(X-x0) + 2*D**2/W**2 * ( # factor of 2 dropped?
                F2(X+x0) - F2(X-x0)
            )
        )
        self.phi = phi
        return self.phi


    def vorticity(self):
        '''
        Eq. (38) from:
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.165433

        Uses mpmath module to compute the infinite sum.
        For 62x496 matrix, this took around 10 minutes.
        '''
        X = self.X
        Y = self.Y
        I = self.I
        n = self.n
        x0 = self.x0
        W = self.W
        D = self.D/W

        if D == 0:
            self.omega = np.zeros(X.shape)
            return self.omega

        def F3(x,y):
            pi = np.pi
            import mpmath
            from mpmath import sin, cos, exp, log, sqrt

            def f(l):
                term1 = (2*l+1)*(-1)**l*cos((2*l+1)*pi*y) \
                    * exp(-abs(x)*sqrt(D**-2+pi**2*(2*l+1)**2))

                term2 = (2*l)*(-1)**l*sin(2*l*pi*y) \
                    * exp(-abs(x)*sqrt(D**-2+pi**2*(2*l)**2))

                return -pi * np.sign(x) * (term1 + term2)
            return mpmath.nsum(f, [0, np.inf])

        f3p = np.full(X.shape, np.nan)
        f3m = np.full(X.shape, np.nan)
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                f3p[j,i] = F3((X[j,i]+x0)/W, Y[j,i]/W)
                f3m[j,i] = F3((X[j,i]-x0)/W, Y[j,i]/W)

        self.omega = 2*I/(n*W**2) * (f3p - f3m) # factor of 2 dropped? I think it's a typo in the paper.
        return self.omega
