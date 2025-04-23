# riemann.py
import numpy as np
from equations import (Advection1D, Advection2D,
                       Burgers1D,  Burgers2D)

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
# 3.  HLL genérico  (sólo requiere c_max)
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
    """Velocidad de onda máxima local (usada para s_L, s_R)."""
    if axis in (None, 0):
        return eq.max_wave_speed_x(U)
    elif axis == 1:
        return eq.max_wave_speed_y(U)
    else:
        raise ValueError("axis debe ser None, 0 o 1")

def riemann_hll(UL, UR, eq, axis=None):
    """
    HLL (dos ondas).  Funciona para cualquier sistema con métodos
    flux_x/flux_y y max_wave_speed_x/_y definidos.
    """
    FL = _flux_by_axis(UL, eq, axis)
    FR = _flux_by_axis(UR, eq, axis)

    aL = _amax_by_axis(UL, eq, axis)
    aR = _amax_by_axis(UR, eq, axis)

    sL = -np.maximum(aL, aR)   # mínima velocidad
    sR =  np.maximum(aL, aR)   # máxima velocidad

    # Tres regiones: sL>0, sL≤0≤sR, sR<0
    F = np.where(sL >= 0, FL,
                 np.where(sR <= 0, FR,
                          (sR*FL - sL*FR + sL*sR*(UR-UL)) / (sR - sL)))
    return F


# ------------------------------------------------------------
# Selector automático
# ------------------------------------------------------------
def solve_riemann(UL, UR, eq, axis=None):
    """
    Envuelve y decide qué solver usar según el tipo de ecuación.
    """
    if isinstance(eq, (Advection1D, Advection2D)):
        return riemann_advection(UL, UR, eq, axis)
    if isinstance(eq, (Burgers1D,  Burgers2D)):
        return riemann_burgers(UL, UR, eq, axis)
    # Genérico por defecto
    return riemann_hll(UL, UR, eq, axis)
