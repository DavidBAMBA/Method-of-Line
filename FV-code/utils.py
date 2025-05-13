# utils.py
import numpy as np
from config import NGHOST   # número de celdas fantasma global

# =============================================================
# 1-D mesh
# =============================================================
def create_mesh_1d(xmin, xmax, Nx, *, return_ghost=False):
    """
    Devuelve la malla de centros de celda 1-D.

    Parameters
    ----------
    xmin, xmax : float
        Extremos del dominio **físico**.
    Nx : int
        Número de celdas físicas.
    return_ghost : bool, optional
        Si True, también devuelve las coordenadas de los centros
        de las NGHOST celdas fantasma a cada lado.

    Returns
    -------
    x_phys : ndarray (Nx,)
        Coordenadas de las celdas físicas.
    dx : float
        Paso de malla.
    x_full : ndarray (Nx+2*NGHOST,), optional
        Coordenadas físicas + fantasmas (solo si return_ghost=True).
    """
    dx = (xmax - xmin) / Nx
    x_phys = np.linspace(xmin + dx/2, xmax - dx/2, Nx)

    if not return_ghost:
        return x_phys, dx

    # Coordenadas para las celdas fantasma (extrapolación uniforme)
    x_left  = x_phys[0]  - dx * np.arange(NGHOST, 0, -1)
    x_right = x_phys[-1] + dx * np.arange(1, NGHOST+1)
    x_full  = np.concatenate([x_left, x_phys, x_right])
    return x_phys, dx, x_full


# =============================================================
# 2-D mesh
# =============================================================
def create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny, *, return_ghost=False):
    """
    Malla 2-D de centros de celda.

    Devuelve (x_phys, y_phys, dx, dy) y, opcionalmente, x_full, y_full
    que incluyen NGHOST celdas fantasma.
    """
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    x_phys = np.linspace(xmin + dx/2, xmax - dx/2, Nx)
    y_phys = np.linspace(ymin + dy/2, ymax - dy/2, Ny)

    if not return_ghost:
        return x_phys, y_phys, dx, dy

    x_left  = x_phys[0]  - dx * np.arange(NGHOST, 0, -1)
    x_right = x_phys[-1] + dx * np.arange(1, NGHOST+1)
    y_bot   = y_phys[0]  - dy * np.arange(NGHOST, 0, -1)
    y_top   = y_phys[-1] + dy * np.arange(1, NGHOST+1)

    x_full = np.concatenate([x_left, x_phys, x_right])
    y_full = np.concatenate([y_bot,  y_phys, y_top])
    return x_phys, y_phys, dx, dy, x_full, y_full


# =============================================================
# U inicial con ghost cells
# =============================================================
def create_U0(nvars, shape_phys, *, initializer=None, ng=NGHOST):
    """
    Crea el array U con ghost cells ya reservadas.

    Parameters
    ----------
    nvars : int
        Número de variables conservadas.
    shape_phys : tuple
        Tamaño del dominio físico (Nx,) o (Nx, Ny).
    initializer : callable(U_phys), optional
        Función que recibe la vista del dominio físico y lo rellena
        in-place.  Ejemplos en initial_conditions.py.
    ng : int
        Número de ghost cells (por defecto NGHOST).

    Returns
    -------
    U : ndarray
        Shape (nvars, Nx+2*ng [, Ny+2*ng]).
    phys_slice : tuple of slice
        Slicing que selecciona únicamente el dominio físico:
        U_phys = U[phys_slice]
    """
    if len(shape_phys) == 1:
        Nx, = shape_phys
        full_shape  = (nvars, Nx + 2*ng)
        phys_slice  = (slice(None), slice(ng, -ng))
    elif len(shape_phys) == 2:
        Nx, Ny = shape_phys
        full_shape = (nvars, Nx + 2*ng, Ny + 2*ng)
        phys_slice = (slice(None), slice(ng, -ng), slice(ng, -ng))
    else:
        raise ValueError("shape_phys debe ser (Nx,) o (Nx, Ny)")

    U = np.zeros(full_shape)

    # Rellenar zona física
    if initializer is not None:
        initializer(U[phys_slice])

    return U, phys_slice