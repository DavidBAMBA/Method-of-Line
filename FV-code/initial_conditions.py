import numpy as np

def gaussian_advection_1d(x, center=0.5, width=0.1):
    def initializer(U):
        U[0, :] = np.exp(-300*(x - center )**2)
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
    x0, y0 = center
    r2 = (X - x0)**2 + (Y - y0)**2
    inside = r2 < radius**2

    rho = np.where(inside, 1.0, 0.125)
    u   = np.where(inside, 0.0, 0.0)
    v   = np.where(inside, 0.0, 0.0)
    p   = np.where(inside, 1.0, 0.1)

    gamma = 1.4
    momx = rho * u
    momy = rho * v
    E    = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    U = np.zeros((4, *X.shape))
    U[0] = rho
    U[1] = momx
    U[2] = momy
    U[3] = E
    return U

def schulz_rinne_2d(X, Y):
    gamma = 1.4
    Nx, Ny = X.shape
    U = np.zeros((4, Nx, Ny))

    Q1 = [1.5, 0.0, 0.0, 1.5]
    Q2 = [33/62, 4/np.sqrt(11), 0.0, 0.3]
    Q3 = [77/558, 4/np.sqrt(11), 4/np.sqrt(11), 9/310]
    Q4 = [33/62, 0.0, 4/np.sqrt(11), 0.3]

    for i in range(Nx):
        for j in range(Ny):
            xi, yj = X[i, j], Y[i, j]
            if xi > 0.8 and yj > 0.8:
                rho, vx, vy, P = Q1
            elif xi <= 0.8 and yj > 0.8:
                rho, vx, vy, P = Q2
            elif xi <= 0.8 and yj <= 0.8:
                rho, vx, vy, P = Q3
            else:
                rho, vx, vy, P = Q4

            U[0, i, j] = rho
            U[1, i, j] = rho * vx
            U[2, i, j] = rho * vy
            U[3, i, j] = P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)
    return U