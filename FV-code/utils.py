import numpy as np

# ------------------------------------------------------------
# Malla 1D sin ghost cells
# ------------------------------------------------------------
def create_mesh_1d(xmin, xmax, Nx):
    """
    Crea malla 1D sin ghost cells.

    Retorna:
        x: coordenadas centradas en celda (Nx,)
        dx: paso de malla
    """
    dx = (xmax - xmin) / Nx
    x = np.linspace(xmin + dx/2, xmax - dx/2, Nx)
    return x, dx


# ------------------------------------------------------------
# Malla 2D sin ghost cells
# ------------------------------------------------------------
def create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny):
    """
    Crea malla 2D sin ghost cells.

    Retorna:
        x, y: mallas 1D en x e y centradas en celda
        dx, dy: pasos de malla
    """
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    x = np.linspace(xmin + dx/2, xmax - dx/2, Nx)
    y = np.linspace(ymin + dy/2, ymax - dy/2, Ny)
    return x, y, dx, dy


# ------------------------------------------------------------
# Inicialización de U sin ghost cells
# ------------------------------------------------------------
def create_U0(nvars, shape, initializer=None):
    """
    Crea arreglo U sin ghost cells.

    Parámetros:
    -----------
    nvars : int
        Número de variables conservadas
    shape : tuple
        (Nx,) o (Nx, Ny)
    initializer : función(U_físico)
        Función que rellena el dominio físico in-place

    Retorna:
    --------
    U : ndarray de shape (nvars, Nx[, Ny])
    """
    U = np.zeros((nvars,) + shape)

    if initializer is not None:
        if len(shape) == 1:
            U[:, :] = initializer(U)
        elif len(shape) == 2:
            U[:, :, :] = initializer(U)
    return U
