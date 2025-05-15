import numpy as np

def gaussian_advection_1d(x, center=0.5, width=0.1):
    def initializer(U):
        U[0, :] = np.exp(-300*(x - center )**2)
        return U
    return initializer

def critical_sine_1d(x):
    """
    IC del test de puntos críticos (Henrick et al. 2005):
        u0(x) = sin(pi*x - sin(pi*x)/pi)
    """
    def initializer(U):
        U[0, :] = np.sin(np.pi*x - np.sin(np.pi*x)/np.pi)
        return U
    return initializer

def sine_advection_1d(x, wavelength=1.0):
    def initializer(U):
        U[0, :] = np.sin(2 * np.pi * x / wavelength)
        return U
    return initializer

def tanh_advection_1d(x, center=0.5, width=0.05):
    def initializer(U):
        U[0, :] = 0.5 * (np.tanh((x - center) / width) + 1.0)
        return U
    return initializer

def square_advection_1d(x, center=0.5, width=0.2):
    def initializer(U):
        U[0, :] = np.where(np.abs(x - center) < width / 2, 1.0, 0.0)
        return U
    return initializer

def gaussian_advection_2d(X, Y, center=(0.5, 0.5), width=0.1, amp=1.0):
    x0, y0 = center
    R2 = (X - x0)**2 + (Y - y0)**2
    return amp * np.exp(-R2 / (2 * width**2))

def gaussian_burgers_1d(x, center=0.5, width=0.1, amp=1.0):
    def initializer(U):
        U[0, :] = amp * np.exp(-((x - center) / width)**2)
        return U
    return initializer

def sinusoidal_burgers_1d(x, *, amp=1.0, phase=0.0):
    k = np.pi / 1.0        # k = 2π/λ,  λ = 2  ⇒ k = π
    def initializer(U):
        U[0, :] = amp * np.sin(k * x + phase)
        return U
    return initializer

def gaussian_burgers_2d(X, Y, center=(0.5, 0.5), width=0.1, amp=1.0):
    x0, y0 = center
    return amp * np.exp(- ((X - x0)**2 + (Y - y0)**2) / width**2)

def sinusoidal_burgers_2d(X, Y, kx=2*np.pi, ky=2*np.pi, amp=1.0):
    return amp * np.sin(kx * X) * np.sin(ky * Y)

def sod_shock_tube_1d(x, x0=0.5):
    def initializer(U):
        rho  = np.where(x < x0, 1.0, 0.125)
        v    = np.where(x < x0, 0.0, 0.0)
        P    = np.where(x < x0, 1.0, 0.1)

        gamma = 1.4
        mom = rho * v
        E = P / (gamma - 1.0) + 0.5 * rho * v**2

        U[0, :] = rho
        U[1, :] = mom
        U[2, :] = E
        return U
    return initializer

def leblanc_shock_tube_1d(x, x0=3.0):
    def initializer(U):
        rho  = np.where(x < x0, 1.0, 1e-3)
        v    = np.zeros_like(x)
        P    = np.where(x < x0, 2/3 * 1e-1, 2/3 * 1e-10)

        gamma = 5.0 / 3.0
        mom = rho * v
        E = P / (gamma - 1.0) + 0.5 * rho * v**2

        U[0, :] = rho
        U[1, :] = mom
        U[2, :] = E
        return U
    return initializer


def sod_shock_tube_2d(x, y, x0=0.5):
    def initializer(U):
        rho = np.where(x < x0, 1.0, 0.125)
        v   = np.where(x < x0, 0.0, 0.0)
        P   = np.where(x < x0, 1.0, 0.1)

        gamma = 1.4
        momx = rho * v
        momy = np.zeros_like(rho)
        E = P / (gamma - 1.0) + 0.5 * rho * v**2

        U[0] = np.repeat(rho[:, None], len(y), axis=1)
        U[1] = np.repeat(momx[:, None], len(y), axis=1)
        U[2] = np.repeat(momy[:, None], len(y), axis=1)
        U[3] = np.repeat(E[:, None], len(y), axis=1)
        return U
    return initializer

def complex_advection_1d(x):
    def initializer(U):
        u = np.zeros_like(x)

        mask1 = (-0.8 <= x) & (x <= -0.6)
        u[mask1] = np.exp(-np.log(2) * (x[mask1] + 0.7)**2 / 0.0009)

        mask2 = (-0.4 <= x) & (x <= -0.2)
        u[mask2] = 1.0

        mask3 = (0.0 <= x) & (x <= 0.2)
        u[mask3] = 1 - np.abs(10 * (x[mask3] - 0.1))

        mask4 = (0.4 <= x) & (x <= 0.6)
        u[mask4] = np.sqrt(1 - 100 * (x[mask4] - 0.5)**2)

        U[0, :] = u
        return U
    return initializer

def shu_osher_1d(x):
    def initializer(U):
        rho = np.where(x < -4, 3.857143, 1 + 0.2 * np.sin(5 * x))
        v   = np.where(x < -4, 2.629369, 0.0)
        P   = np.where(x < -4, 10.3333, 1.0)

        gamma = 1.4
        mom = rho * v
        E   = P / (gamma - 1.0) + 0.5 * rho * v**2

        U[0, :] = rho
        U[1, :] = mom
        U[2, :] = E
        return U
    return initializer

def explosion_problem_2d(X, Y, center=(1.0, 1.0), radius=0.4):
    def initializer(U):
        x0, y0 = center
        r2 = (X - x0)**2 + (Y - y0)**2
        inside = r2 < radius**2

        gamma = 1.4

        # Condiciones iniciales 2D usando máscaras
        rho = np.where(inside, 1.0, 0.125)
        v   = np.zeros_like(rho)
        p   = np.where(inside, 1.0, 0.1)

        momx = rho * v
        momy = rho * v
        E    = p / (gamma - 1.0) + 0.5 * rho * v**2

        U[0] = rho
        U[1] = momx
        U[2] = momy
        U[3] = E
        return U
    return initializer



def schulz_rinne_2d(X, Y):
    """
    Inicializador vectorizado para el test de Schulz-Rinne (1993).
    Divide el dominio en 4 cuadrantes y asigna condiciones iniciales distintas.
    """
    def initializer(U):
        gamma = 1.4

        # Condiciones por cuadrante
        Q1 = [1.5,      0.0,             0.0,             1.5]
        Q2 = [33/62,    4/np.sqrt(11),   0.0,             0.3]
        Q3 = [77/558,   4/np.sqrt(11),   4/np.sqrt(11),   9/310]
        Q4 = [33/62,    0.0,             4/np.sqrt(11),   0.3]

        # Máscaras para cada región
        m1 = (X > 0.8) & (Y > 0.8)
        m2 = (X <= 0.8) & (Y > 0.8)
        m3 = (X <= 0.8) & (Y <= 0.8)
        m4 = (X > 0.8) & (Y <= 0.8)

        # Inicializar arrays
        rho = np.zeros_like(X)
        vx  = np.zeros_like(X)
        vy  = np.zeros_like(X)
        P   = np.zeros_like(X)

        # Asignación vectorizada
        for mask, (r, vx_, vy_, p) in zip([m1, m2, m3, m4], [Q1, Q2, Q3, Q4]):
            rho[mask] = r
            vx[mask]  = vx_
            vy[mask]  = vy_
            P[mask]   = p

        # Calcular variables conservadas
        momx = rho * vx
        momy = rho * vy
        E    = P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)

        # Asignar al arreglo U
        U[0] = rho
        U[1] = momx
        U[2] = momy
        U[3] = E

        return U
    return initializer

