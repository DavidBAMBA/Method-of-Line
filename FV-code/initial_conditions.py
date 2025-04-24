import numpy as np

# -----------------------------
# Advección escalar
# -----------------------------

def gaussian_advection_1d(x, center=0.5, width=0.1):
    """
    Pulso gaussiano para advección escalar 1D.
    """
    def initializer(U):
        U[0] = np.exp(-((x - center) / width)**2)
        return U
    return initializer


def gaussian_advection_2d(X, Y, center=(0.5, 0.5), width=0.1, amp=1.0):
    """
    Pulso gaussiano centrado en (x0, y0).
    
    Parámetros:
    -----------
    X, Y : mallas 2D (Nx, Ny)
    center : tupla (x0, y0)
    width : sigma del gaussiano
    amp : amplitud máxima

    Retorna:
    --------
    U : array 2D (Nx, Ny)
    """
    x0, y0 = center
    R2 = (X - x0)**2 + (Y - y0)**2
    return amp * np.exp(-R2 / (2 * width**2))



# -----------------------------
# Burgers escalar
# -----------------------------

def gaussian_burgers_1d(x, center=0.5, width=0.1, amp=1.0):
    """
    Pulso gaussiano para Burgers escalar 1D.
    """
    def initializer(U):
        U[0] = amp * np.exp(-((x - center) / width)**2)
        return U
    return initializer


import numpy as np

def gaussian_burgers_2d(X, Y, center=(0.5, 0.5), width=0.1, amp=1.0):
    """
    Pulso gaussiano escalar para Burgers 2D.

    Parámetros:
    -----------
    X, Y : mallas 2D (Nx, Ny) con np.meshgrid
    center : tupla (x0, y0)
    width : sigma del gaussiano
    amp : amplitud máxima

    Retorna:
    --------
    ndarray de shape (Nx, Ny) con el pulso gaussiano
    """
    x0, y0 = center
    return amp * np.exp(- ((X - x0)**2 + (Y - y0)**2) / width**2)

import numpy as np

def sinusoidal_burgers_2d(X, Y, kx=2*np.pi, ky=2*np.pi, amp=1.0):
    """
    Perfil senoidal periódico para Burgers 2D.

    Parámetros:
    -----------
    X, Y : mallas 2D (Nx, Ny) generadas con np.meshgrid
    kx, ky : número de ondas (frecuencia angular) en x e y
    amp : amplitud máxima

    Retorna:
    --------
    ndarray de shape (Nx, Ny) con perfil u(x,y,0) = A sin(kx x) sin(ky y)
    """
    return amp * np.sin(kx * X) * np.sin(ky * Y)


# -----------------------------
# Euler 1D – Sod shock tube
# -----------------------------

def sod_shock_tube_1d(x, x0=0.5):
    """
    Condición inicial clásica de Sod para Euler 1D.
    Regiones izquierda/derecha con salto en x0.
    """
    def initializer(U):
        rho  = np.where(x < x0, 1.0, 0.125)
        v    = np.where(x < x0, 0.0, 0.0)
        P    = np.where(x < x0, 1.0, 0.1)
        
        gamma = 1.4
        mom = rho * v
        E = P / (gamma - 1.0) + 0.5 * rho * v**2
        
        U[0] = rho
        U[1] = mom
        U[2] = E
        return U
    return initializer


def complex_advection_1d(x):
    """
    Onda compuesta para la ecuación de advección lineal.
    Combina: cuadrado, gaussiana, triángulo y semi-elipse.
    """
    def initializer(U):
        u = np.zeros_like(x)

        # Gaussiana en [-0.8, -0.6]
        mask1 = (-0.8 <= x) & (x <= -0.6)
        u[mask1] = np.exp(-np.log(2) * (x[mask1] + 0.7)**2 / 0.0009)

        # Triángulo en [-0.4, -0.2]
        mask2 = (-0.4 <= x) & (x <= -0.2)
        u[mask2] = 1 - np.abs((x[mask2] + 0.3) / 0.1)

        # Cuadrado en [0, 0.2]
        mask3 = (0.0 <= x) & (x <= 0.2)
        u[mask3] = 1.0

        # Semi-elipse en [0.4, 0.6]
        mask4 = (0.4 <= x) & (x <= 0.6)
        u[mask4] = np.sqrt(1 - 100 * (x[mask4] - 0.5)**2)

        U[0] = u
        return U
    return initializer


def shu_osher_1d(x):
    """
    Condición inicial para el problema de Shu–Osher.
    Dominio típico: x ∈ [-5, 5]
    """
    def initializer(U):
        rho = np.where(x < -4, 3.857143, 1 + 0.2 * np.sin(5 * x))
        v   = np.where(x < -4, 2.629369, 0.0)
        P   = np.where(x < -4, 10.3333, 1.0)

        gamma = 1.4
        mom = rho * v
        E   = P / (gamma - 1.0) + 0.5 * rho * v**2

        U[0] = rho
        U[1] = mom
        U[2] = E
        return U
    return initializer


def explosion_problem_2d(X, Y, center=(1.0, 1.0), radius=0.4):
    """
    Condiciones iniciales para el problema de explosión 2D.

    Parámetros:
    -----------
    X, Y : mallas 2D con shape (Nx, Ny)
    center : coordenadas del centro de la explosión
    radius : radio de la región caliente

    Retorna:
    --------
    U : ndarray (4, Nx, Ny) con [ρ, ρu, ρv, E]
    """
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


import numpy as np

def schulz_rinne_2d(x, y):
    """
    Problema de Riemann 2D (Schulz–Rinne et al. config 3) en dominio [0,1]×[0,1].
    Inicializa las variables (ρ, ρv_x, ρv_y, E) en cada cuadrante.

    Parámetros:
    -----------
    x, y : arreglos 1D de coordenadas

    Retorna:
    --------
    U : ndarray shape (4, Nx, Ny)
    """
    gamma = 1.4
    X, Y = np.meshgrid(x, y, indexing='ij')
    Nx, Ny = len(x), len(y)
    U = np.zeros((4, Nx, Ny))

    # Estados por cuadrante
    # [rho, vx, vy, P]
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
            else:  # xi > 0.8 and yj <= 0.8
                rho, vx, vy, P = Q4

            U[0, i, j] = rho
            U[1, i, j] = rho * vx
            U[2, i, j] = rho * vy
            U[3, i, j] = P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2)

    return U


