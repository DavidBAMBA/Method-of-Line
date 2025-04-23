import numpy as np

# ------------------- Limitadores -------------------

def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod3(a, b, c):
    cond1 = (a > 0) & (b > 0) & (c > 0)
    cond2 = (a < 0) & (b < 0) & (c < 0)
    result = np.where(cond1, np.minimum(np.minimum(a, b), c), 0.0)
    result = np.where(cond2, np.maximum(np.maximum(a, b), c), result)
    return result

def maxmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) > np.abs(b), a, b), 0.0)

# ------------------- Slopes -------------------

def slope_minmod(U, dx):
    return minmod((U[1:-1] - U[:-2]) / dx,
                  (U[2:]   - U[1:-1]) / dx)

def slope_mc(U, dx):
    a = (U[2:] - U[:-2]) / (2 * dx)
    b = 2 * (U[1:-1] - U[:-2]) / dx
    c = 2 * (U[2:]   - U[1:-1]) / dx
    return minmod3(a, b, c)

def slope_superbee(U, dx):
    s1 = minmod(2 * (U[1:-1] - U[:-2]) / dx,
                (U[2:]   - U[1:-1]) / dx)
    s2 = minmod((U[1:-1] - U[:-2]) / dx,
                2 * (U[2:] - U[1:-1]) / dx)
    return maxmod(s1, s2)

# ------------------- Reconstrucción -------------------

def reconstruct(U, dx, limiter="minmod", axis=None):
    """
    Reconstrucción MUSCL lineal sin ghost cells.
    Devuelve Nx+1 o Ny+1 interfaces según axis:
    - UL[i] = U[i] + 0.5 Δ_i
    - UR[i] = U[i+1] - 0.5 Δ_{i+1}

    Parámetros:
    -----------
    U : ndarray
        (nvars, Nx) o (nvars, Nx, Ny)
    dx : float
    limiter : "minmod", "mc", "superbee"
    axis : None (1D), 0 (x), 1 (y)

    Retorna:
    --------
    UL, UR : arrays con nvars y Nx+1 / Ny+1 interfaces
    """

    def compute_slope(Uk):
        Nx = Uk.shape[0]
        dqm = Uk[1:] - Uk[:-1]
        dqp = np.empty_like(dqm)
        dqp[:-1] = dqm[1:]
        dqp[-1] = dqm[-1]

        if limiter == "minmod":
            slope = minmod(dqm, dqp)
        elif limiter == "mc":
            a = (np.roll(Uk, -1) - np.roll(Uk, 1)) / (2 * dx)
            b = 2 * dqm / dx
            c = 2 * dqp / dx
            slope = minmod3(a[:-1], b, c)
        elif limiter == "superbee":
            s1 = minmod(2 * dqm / dx, dqp / dx)
            s2 = minmod(dqm / dx, 2 * dqp / dx)
            slope = maxmod(s1, s2)
        else:
            raise ValueError(f"Unknown limiter: {limiter}")

        return np.append(slope, slope[-1])

    # ----------- 1D -------------
    if axis is None:
        nvars, Nx = U.shape
        UL = np.zeros((nvars, Nx + 1))
        UR = np.zeros((nvars, Nx + 1))

        for k in range(nvars):
            Uk = U[k]
            slope = compute_slope(Uk)
            UL[k, 1:] = Uk + 0.5 * dx * slope
            UR[k, :-1] = Uk - 0.5 * dx * slope
            UL[k, 0] = Uk[0]
            UR[k, -1] = Uk[-1]

        return UL, UR

    # ----------- 2D: Dirección x -------------
    elif axis == 0:
        nvars, Nx, Ny = U.shape
        UL = np.zeros((nvars, Nx + 1, Ny))
        UR = np.zeros((nvars, Nx + 1, Ny))

        for k in range(nvars):
            for j in range(Ny):
                Uk = U[k, :, j]
                slope = compute_slope(Uk)
                UL[k, 1:, j] = Uk + 0.5 * dx * slope
                UR[k, :-1, j] = Uk - 0.5 * dx * slope
                UL[k, 0, j] = Uk[0]
                UR[k, -1, j] = Uk[-1]

        return UL, UR

    # ----------- 2D: Dirección y -------------
    elif axis == 1:
        nvars, Nx, Ny = U.shape
        UL = np.zeros((nvars, Nx, Ny + 1))
        UR = np.zeros((nvars, Nx, Ny + 1))

        for k in range(nvars):
            for i in range(Nx):
                Uk = U[k, i, :]
                slope = compute_slope(Uk)
                UL[k, i, 1:] = Uk + 0.5 * dx * slope
                UR[k, i, :-1] = Uk - 0.5 * dx * slope
                UL[k, i, 0] = Uk[0]
                UR[k, i, -1] = Uk[-1]

        return UL, UR

    else:
        raise ValueError("Axis must be None (1D), 0 (x), or 1 (y)")
