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
