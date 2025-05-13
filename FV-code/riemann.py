# riemann.py
import numpy as np
from equations import (Advection1D, Advection2D,
                       Burgers1D,  Burgers2D, Euler1D, Euler2D)

# ------------------------------------------------------------
# 1.  Advección escalar  (exacto)
# ------------------------------------------------------------
def riemann_advection(UL, UR, eq, axis=None):
    """
    Exacto para  U_t + a U_x = 0  (o U_t + a U_y = 0).
    Usa dirección 'axis'  (None|0|1).
    """
    a = eq.ax if axis in (None, 0) else eq.ay
    # Signo de a decide el flujo upwind
    return a * (UL if a >= 0 else UR)


# ------------------------------------------------------------
# 2.  Burgers escalar  (exacto)
# ------------------------------------------------------------
def riemann_burgers(UL, UR, *_):
    """
    Exacto para Burgers escalar.
    UL, UR con shape (nvars=1, ...); vectorizado ≥1 D.
    """
    FL = 0.5 * UL**2
    FR = 0.5 * UR**2
    F  = np.zeros_like(UL)

    # Shock   → max
    mask_shock = UL > UR
    F[mask_shock] = np.maximum(FL[mask_shock], FR[mask_shock])

    # Rarefaction → min
    mask_rare = ~mask_shock
    F[mask_rare] = np.minimum(FL[mask_rare], FR[mask_rare])

    # Ráfaga que cruza el 0 → 0
    fan = (UL <= 0) & (UR > 0)
    F[fan] = 0.0
    return F


# ------------------------------------------------------------
# 3.  HLL genérico 
# ------------------------------------------------------------

def _flux_by_axis(U, eq, axis):
    """Devuelve el flujo (nvars, …) según axis."""
    if axis in (None, 0):
        return eq.flux_x(U)
    elif axis == 1:
        return eq.flux_y(U)
    else:
        raise ValueError("axis debe ser None, 0 o 1")

def _amax_by_axis(U, eq, axis):
    """Velocidad de onda máxima local (sonido)."""
    if axis in (None, 0):
        return eq.max_wave_speed_x(U)
    elif axis == 1:
        return eq.max_wave_speed_y(U)
    else:
        raise ValueError("axis debe ser None, 0 o 1")

def _velocity_by_axis(U, axis):
    rho = U[0]
    if axis in (None, 0):
        rhov = U[1]
    elif axis == 1:
        rhov = U[2]
    else:
        raise ValueError("axis debe ser None, 0 o 1")
    return rhov / rho

def riemann_hll(UL, UR, eq, axis=None):
    """
    HLL para sistemas tipo Euler en 1D o 2D.
    """
    FL = _flux_by_axis(UL, eq, axis)
    FR = _flux_by_axis(UR, eq, axis)

    vL = _velocity_by_axis(UL, axis)
    vR = _velocity_by_axis(UR, axis)

    cL = _amax_by_axis(UL, eq, axis)
    cR = _amax_by_axis(UR, eq, axis)

    sL = np.minimum(vL - cL, vR - cR)
    sR = np.maximum(vL + cL, vR + cR)

    denom = sR - sL
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    F = np.where(sL >= 0, FL,
                 np.where(sR <= 0, FR,
                          (sR * FL - sL * FR + sL * sR * (UR - UL)) / denom))
    return F

# ------------------------------------------------------------
# 4.  HLLC (tres ondas)
# ------------------------------------------------------------
def riemann_hllc(UL, UR, eq, axis=None):
    """
    HLLC adaptable para 1D (Euler1D) y 2D (Euler2D). Basado en Batten et al.
    """
    FL = _flux_by_axis(UL, eq, axis)
    FR = _flux_by_axis(UR, eq, axis)

    rhoL, rhoR = UL[0], UR[0]
    gamma = getattr(eq, "gamma", 1.4)

    if UL.shape[0] == 3:  # ← Euler1D
        momL, momR = UL[1], UR[1]
        vL = momL / rhoL
        vR = momR / rhoR
        pL = eq._pressure(UL)
        pR = eq._pressure(UR)

        cL = np.sqrt(gamma * pL / rhoL)
        cR = np.sqrt(gamma * pR / rhoR)

        sL = np.minimum(vL - cL, vR - cR)
        sR = np.maximum(vL + cL, vR + cR)

        denom = rhoL * (sL - vL) - rhoR * (sR - vR)
        sM_num = pR - pL + rhoL * vL * (sL - vL) - rhoR * vR * (sR - vR)
        sM = np.where(np.abs(denom) > 1e-12, sM_num / denom, 0.5 * (vL + vR))

        rho_star_L = rhoL * (sL - vL) / (sL - sM)
        rho_star_R = rhoR * (sR - vR) / (sR - sM)
        p_star = pL + rhoL * (sL - vL) * (sM - vL)

        E_star_L = p_star / (gamma - 1) + 0.5 * rho_star_L * sM**2
        E_star_R = p_star / (gamma - 1) + 0.5 * rho_star_R * sM**2

        U_star_L = np.array([rho_star_L,
                             rho_star_L * sM,
                             E_star_L])
        U_star_R = np.array([rho_star_R,
                             rho_star_R * sM,
                             E_star_R])

    elif UL.shape[0] == 4:  # ← Euler2D
        if axis in (None, 0):  # eje x
            uL_n, uR_n = UL[1] / rhoL, UR[1] / rhoR
            uL_t, uR_t = UL[2] / rhoL, UR[2] / rhoR
        else:  # eje y
            uL_n, uR_n = UL[2] / rhoL, UR[2] / rhoR
            uL_t, uR_t = UL[1] / rhoL, UR[1] / rhoR

        pL = eq._pressure(UL)
        pR = eq._pressure(UR)
        cL = np.sqrt(gamma * pL / rhoL)
        cR = np.sqrt(gamma * pR / rhoR)

        sL = np.minimum(uL_n - cL, uR_n - cR)
        sR = np.maximum(uL_n + cL, uR_n + cR)

        denom = rhoL * (sL - uL_n) - rhoR * (sR - uR_n)
        sM = (pR - pL + rhoL * uL_n * (sL - uL_n) - rhoR * uR_n * (sR - uR_n))
        sM = np.where(denom != 0, sM / denom, 0.5 * (uL_n + uR_n))

        rho_star_L = rhoL * (sL - uL_n) / (sL - sM)
        rho_star_R = rhoR * (sR - uR_n) / (sR - sM)
        p_star = pL + rhoL * (sL - uL_n) * (sM - uL_n)

        E_star_L = p_star / (gamma - 1) + 0.5 * rho_star_L * (sM**2 + uL_t**2)
        E_star_R = p_star / (gamma - 1) + 0.5 * rho_star_R * (sM**2 + uR_t**2)

        U_star_L = np.empty_like(UL)
        U_star_R = np.empty_like(UR)
        U_star_L[0] = rho_star_L
        U_star_R[0] = rho_star_R
        if axis in (None, 0):
            U_star_L[1] = rho_star_L * sM
            U_star_R[1] = rho_star_R * sM
            U_star_L[2] = rho_star_L * uL_t
            U_star_R[2] = rho_star_R * uR_t
        else:
            U_star_L[1] = rho_star_L * uL_t
            U_star_R[1] = rho_star_R * uR_t
            U_star_L[2] = rho_star_L * sM
            U_star_R[2] = rho_star_R * sM
        U_star_L[3] = E_star_L
        U_star_R[3] = E_star_R

    else:
        raise ValueError("Formato de estado U no reconocido para riemann_hllc")

    F_star_L = FL + sL * (U_star_L - UL)
    F_star_R = FR + sR * (U_star_R - UR)

    # broadcasting
    sL_ext = sL[np.newaxis, ...]
    sR_ext = sR[np.newaxis, ...]
    sM_ext = sM[np.newaxis, ...]

    F = np.where(sL_ext > 0, FL,
                 np.where(sR_ext < 0, FR,
                          np.where(sM_ext >= 0, F_star_L, F_star_R)))
    return F

# Actualizar el selector automático para incluir HLLC
def solve_riemann(UL, UR, eq, axis=None, solver="hllc"):
    """
    Selector de solver de Riemann.

    Parámetros:
    ------------
    UL, UR : Estados izquierdo y derecho (shape (nvars, Nx[, Ny]))
    eq     : Objeto de ecuaciones (Advection, Burgers, Euler, etc.)
    axis   : None (1D), 0 (x) o 1 (y)
    solver : "hll" o "hllc" (sólo afecta a Euler)

    Retorna:
    --------
    F : Flujo numérico en las interfaces
    """
    if isinstance(eq, (Advection1D, Advection2D)):
        return riemann_advection(UL, UR, eq, axis)
    if isinstance(eq, (Burgers1D, Burgers2D)):
        return riemann_burgers(UL, UR, eq, axis)
    if isinstance(eq, Euler2D) or isinstance(eq, Euler1D):
        if solver.lower() == "hllc":
            return riemann_hllc(UL, UR, eq, axis)
        elif solver.lower() == "hll":
            return riemann_hll(UL, UR, eq, axis)
        else:
            raise ValueError(f"Solver '{solver}' no reconocido para Euler (usa 'hll' o 'hllc').")
    # Genérico por defecto
    return riemann_hll(UL, UR, eq, axis)
