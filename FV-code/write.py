# write.py
"""
Salida de resultados en formato CSV.

*  Trabaja con arreglos **que incluyen ghost cells**.
*  Sólo se escribe la zona física (Nx x Ny).
*  Soporta:
     -Escalar (advección, Burgers)  nvars = 1
     -Euler 1-D  nvars = 3   [rho, rhov,   E]
     -Euler 2-D  nvars = 4   [rho, rhovx, rhovy, E]
"""
import os
import numpy as np
from config import NGHOST


# ------------------------------------------------------------------
#  Rutinas auxiliares
# ------------------------------------------------------------------
def _primitive_1d(rho, mom, E, gamma):
    v = mom / rho
    P = (gamma - 1.0) * (E - 0.5 * rho * v**2)
    return v, P

def _primitive_2d(rho, momx, momy, E, gamma):
    vx, vy = momx / rho, momy / rho
    P = (gamma - 1.0) * (E - 0.5 * rho * (vx**2 + vy**2))
    return vx, vy, P

# ------------------------------------------------------------------
def setup_data_folder(path="data"):
    os.makedirs(path, exist_ok=True)

def save_all_fields(U, x, y,
                    step,
                    gamma=1.4,
                    path="data",
                    prefix="output",
                    reconstructor="none",
                    time=None):
    """
    Guarda CSV con las variables más relevantes.

    Parameters
    ----------
    U : ndarray
        Shape (nvars, Nx+2*NGHOST [, Ny+2*NGHOST])
    x, y : 1-D arrays con las coordenadas **físicas** (sin fantasmas).
    step : int
    gamma : float
    path  : str
    prefix, reconstructor : str
    time : float | None
    """
    g = NGHOST
    nvars = U.shape[0]

    # --------------- 1-D ------------------------------------------
    if U.ndim == 2:
        Up = U[:, g:-g]                    # (nvars, Nx)
        Nx = Up.shape[1]

        if nvars == 1:                     # Escalar
            cols = [x, Up[0]]
            header = "x, u"
        elif nvars == 3:                   # Euler 1-D
            rho, mom, E = Up
            v, P = _primitive_1d(rho, mom, E, gamma)
            cols = [x, rho, v, P]
            header = "x, rho, v, P"
        else:
            raise ValueError("save_all_fields: nvars incompatible para 1D")

        if time is not None:
            cols.append(np.full(Nx, time))
            header += ", time"

        data = np.column_stack(cols)

    # --------------- 2-D ------------------------------------------
    elif U.ndim == 3:
        Up = U[:, g:-g, g:-g]              # (nvars, Nx, Ny)
        Nx, Ny = Up.shape[1:]

        X, Y = np.meshgrid(x, y, indexing='ij')

        if nvars == 1:                     # Escalar 2-D
            cols = [X.ravel(), Y.ravel(), Up[0].ravel()]
            header = "x, y, u"
        elif nvars == 4:                   # Euler 2-D
            rho, momx, momy, E = Up
            vx, vy, P = _primitive_2d(rho, momx, momy, E, gamma)
            cols = [X.ravel(), Y.ravel(),
                    rho.ravel(), vx.ravel(), vy.ravel(), P.ravel()]
            header = "x, y, rho, vx, vy, P"
        else:
            raise ValueError("save_all_fields: nvars incompatible para 2D")

        if time is not None:
            cols.append(np.full(X.size, time))
            header += ", time"

        data = np.column_stack(cols)

    else:
        raise ValueError("U debe tener 2 (1D) o 3 (2D) dimensiones.")

    # --------------- escritura ------------------------------------
    setup_data_folder(path)
    fname = f"{prefix}_{reconstructor}_{step:05d}.csv"
    fullpath = os.path.join(path, fname)
    np.savetxt(fullpath, data, delimiter=",", header=header, comments='')