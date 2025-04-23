# boundary.py – versión sin ghost cells
import numpy as np

def periodic(U):
    """
    Aplica condiciones periódicas sobrescribiendo las dos celdas de borde.
    U: ndarray (nvars, Nx) o (nvars, Nx, Ny)
    """
    if U.ndim == 2:  # 1D
        U[:, 0]  = U[:, -2]
        U[:, -1] = U[:, 1]
    elif U.ndim == 3:  # 2D
        # Dirección x (bordes izquierdo y derecho)
        U[:, 0, :]  = U[:, -2, :]
        U[:, -1, :] = U[:, 1, :]

        # Dirección y (bordes inferior y superior)
        U[:, :, 0]  = U[:, :, -2]
        U[:, :, -1] = U[:, :, 1]
    else:
        raise ValueError("U debe tener 2 o 3 dimensiones.")
    return U

import numpy as np

def dirichlet_sod(U):
    """
    Condiciones fijas de Sod.
    • 1-D  →  (ρ, ρv,  E)      (3 vars)
    • 2-D  →  (ρ, ρv_x, ρv_y, E) (4 vars)
    """
    if U.shape[0] == 3:                 # --- 1-D ---
        left  =  np.array([1.0  , 0.0 , 2.5 ])
        right =  np.array([0.125, 0.0 , 0.25])
    elif U.shape[0] == 4:               # --- 2-D ---
        left  =  np.array([1.0  , 0.0 , 0.0, 2.5 ])
        right =  np.array([0.125, 0.0 , 0.0, 0.25])
    else:
        raise ValueError("dirichlet_sod: nvars debe ser 3 ó 4")

    if U.ndim == 2:            # 1-D
        U[:, 0]  = left
        U[:, -1] = right
    elif U.ndim == 3:          # 2-D
        U[:, 0, :]  = left[:,  None]
        U[:, -1, :] = right[:, None]
    else:
        raise ValueError("U debe tener 2 o 3 dimensiones.")
    return U




def dirichlet(U, value=0.0):
    """
    Impone valor constante en las celdas de borde.
    """
    if U.ndim == 2:
        U[:, 0]  = value
        U[:, -1] = value
    elif U.ndim == 3:
        U[:, 0, :]  = value
        U[:, -1, :] = value
        U[:, :, 0]  = value
        U[:, :, -1] = value
    else:
        raise ValueError("U debe tener 2 o 3 dimensiones.")
    return U


def extrapolate(U):
    """
    Celdas de borde duplican el valor interior contiguo (outflow).
    """
    if U.ndim == 2:
        U[:, 0]  = U[:, 1]
        U[:, -1] = U[:, -2]
    elif U.ndim == 3:
        # X
        U[:, 0, :]  = U[:, 1, :]
        U[:, -1, :] = U[:, -2, :]
        # Y
        U[:, :, 0]  = U[:, :, 1]
        U[:, :, -1] = U[:, :, -2]
    else:
        raise ValueError("U debe tener 2 o 3 dimensiones.")
    return U
