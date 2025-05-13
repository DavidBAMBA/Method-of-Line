""" import numpy as np
from config import NGHOST

# ------------------------------------------------------------------
#  Limitadores elementales
# ------------------------------------------------------------------
def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod3(a, b, c):
    cond_pos = (a > 0) & (b > 0) & (c > 0)
    cond_neg = (a < 0) & (b < 0) & (c < 0)
    res = np.where(cond_pos, np.minimum(np.minimum(a, b), c), 0.0)
    res = np.where(cond_neg, np.maximum(np.maximum(a, b), c), res)
    return res

def maxmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) > np.abs(b), a, b), 0.0)

# ------------------------------------------------------------------
#  Utilidades MP5
# ------------------------------------------------------------------
def minmod_pair(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod4(w, x, y, z):
    s = 0.125 * (np.sign(w) + np.sign(x)) * np.abs((np.sign(w) + np.sign(y)) * (np.sign(w) + np.sign(z)))
    res = s * np.min(np.stack([np.abs(w), np.abs(x), np.abs(y), np.abs(z)]), axis=0)
    res[(w == 0) | (x == 0) | (y == 0) | (z == 0)] = 0.0
    return res

# ------------------------------------------------------------------
#  Pendientes MUSCL (versión menos disipativa)
# ------------------------------------------------------------------
def slope_full(U, dx, limiter="minmod"):

    s_inner = None
    if limiter == "minmod":
        a = (U[1:-1] - U[:-2]) / dx
        b = (U[2:]   - U[1:-1]) / dx
        s_inner = minmod(a, b)
    elif limiter == "mc":
        a = (U[2:] - U[:-2]) / (2 * dx)
        b = 2 * (U[1:-1] - U[:-2]) / dx
        c = 2 * (U[2:] - U[1:-1]) / dx
        s_inner = minmod3(a, b, c)
    elif limiter == "superbee":
        a = 2 * (U[1:-1] - U[:-2]) / dx
        b = (U[2:] - U[1:-1]) / dx
        s1 = minmod(a, b)
        a = (U[1:-1] - U[:-2]) / dx
        b = 2 * (U[2:] - U[1:-1]) / dx
        s2 = minmod(a, b)
        s_inner = maxmod(s1, s2)
    else:
        raise ValueError(f"Limitador '{limiter}' no reconocido.")

    slope = np.empty_like(U)
    slope[1:-1] = s_inner
    slope[0] = s_inner[0]
    slope[-1] = s_inner[-1]
    return slope

# ------------------------------------------------------------------
#  MP5 (requiere ≥2 ghost)
# ------------------------------------------------------------------
def mp5_faces(U, dx, alpha=4.0, eps=1e-6):
    if NGHOST < 2:
        raise ValueError("MP5 requiere al menos 2 ghost cells")

    um2, um1, u0, up1, up2 = U[:-4], U[1:-3], U[2:-2], U[3:-1], U[4:]
    vnorm = np.sqrt(um2**2 + um1**2 + u0**2 + up1**2 + up2**2)

    vl = (2*um2 - 13*um1 + 47*u0 + 27*up1 - 3*up2) / 60.0
    vmp = u0 + minmod_pair(up1 - u0, alpha*(u0 - um1))
    smooth = (vl - u0)*(vl - vmp) <= eps*vnorm

    djm1 = um2 - 2*um1 + u0
    dj   = um1 - 2*u0 + up1
    djp1 = u0  - 2*up1 + up2
    dm4jmh = minmod4(4*dj - djm1, 4*djm1 - dj, dj, djm1)
    dm4jph = minmod4(4*dj - djp1, 4*djp1 - dj, dj, djp1)

    vul = u0 + alpha*(u0 - um1)
    vav = 0.5*(u0 + up1)
    vmd = vav - 0.5*dm4jph
    vlc = u0 + 0.5*(u0 - um1) + (4/3)*dm4jmh
    vminl = np.maximum(np.minimum.reduce([u0, up1, vmd]),
                       np.minimum.reduce([u0, vul, vlc]))
    vmaxl = np.minimum(np.maximum.reduce([u0, up1, vmd]),
                       np.maximum.reduce([u0, vul, vlc]))
    vl += minmod_pair(vminl - vl, vmaxl - vl)
    vl = np.where(smooth, vl, vl)

    vr = (2*up2 - 13*up1 + 47*u0 + 27*um1 - 3*um2) / 60.0
    vmp_r = u0 + minmod_pair(um1 - u0, alpha*(u0 - up1))
    smooth_r = (vr - u0)*(vr - vmp_r) <= eps*vnorm

    djm1_r = up2 - 2*up1 + u0
    dj_r   = up1 - 2*u0 + um1
    djp1_r = u0  - 2*um1 + um2
    dm4jmh_r = minmod4(4*dj_r - djm1_r, 4*djm1_r - dj_r, dj_r, djm1_r)
    dm4jph_r = minmod4(4*dj_r - djp1_r, 4*djp1_r - dj_r, dj_r, djp1_r)

    vul_r = u0 + alpha*(u0 - up1)
    vav_r = 0.5*(u0 + um1)
    vmd_r = vav_r - 0.5*dm4jph_r
    vlc_r = u0 + 0.5*(u0 - up1) + (4/3)*dm4jmh_r
    vminr = np.maximum(np.minimum.reduce([u0, um1, vmd_r]),
                       np.minimum.reduce([u0, vul_r, vlc_r]))
    vmaxr = np.minimum(np.maximum.reduce([u0, um1, vmd_r]),
                       np.maximum.reduce([u0, vul_r, vlc_r]))
    vr += minmod_pair(vminr - vr, vmaxr - vr)
    vr = np.where(smooth_r, vr, vr)
    return vl, vr

# ------------------------------------------------------------------
#  Reconstrucción general
# ------------------------------------------------------------------
def reconstruct(U, dx, *, limiter="minmod", axis=None):
    g = NGHOST

    if axis is None:
        nvars, Ntot = U.shape
        Nx = Ntot - 2*g
        UL = np.zeros((nvars, Nx+1))
        UR = np.zeros((nvars, Nx+1))

        for k in range(nvars):
            Uk = U[k]

            if limiter == "mp5":
                vl, vr = mp5_faces(Uk, dx)
                if g > 2:
                    cut = g - 2
                    vl, vr = vl[cut:-cut], vr[cut:-cut]
                UL[k, 1:]  = vl
                UR[k, :-1] = vr
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:]  = Uk[g:-g] + 0.5*dx*slope
                UR[k, :-1] = Uk[g:-g] - 0.5*dx*slope
                #UL[k, 1:Nx] = Uk[g:g+Nx-1] + 0.5*dx*slope[:-1]   # longitud Nx-1 → OK
                #UR[k, 1:Nx] = Uk[g+1:g+Nx]  - 0.5*dx*slope[1:]

            UL[k, 0]  = Uk[g-1]
            UR[k, 0]  = Uk[g]
            UL[k, -1] = Uk[g+Nx-1]
            UR[k, -1] = Uk[g+Nx]

        return UL, UR

    if axis in (0, 1):
        swap = (axis == 1)
        if swap:
            U = U.swapaxes(1, 2)

        nvars, Nx_tot, Ny = U.shape
        Nx = Nx_tot - 2*g
        UL = np.zeros((nvars, Nx+1, Ny))
        UR = np.zeros((nvars, Nx+1, Ny))

        for k in range(nvars):
            for j in range(Ny):
                Uk = U[k, :, j]

                if limiter == "mp5":
                    vl, vr = mp5_faces(Uk, dx)
                    if g > 2:
                        cut = g - 2
                        vl, vr = vl[cut:-cut], vr[cut:-cut]
                    UL[k, 1:, j]  = vl
                    UR[k, :-1, j] = vr
                else:
                    slope = slope_full(Uk[g:-g], dx, limiter)
                    UL[k, 1:, j]  = Uk[g:-g] + 0.5*dx*slope
                    UR[k, :-1, j] = Uk[g:-g] - 0.5*dx*slope

                UL[k, 0,  j] = Uk[g-1]
                UR[k, 0,  j] = Uk[g]
                UL[k, -1, j] = Uk[g+Nx-1]
                UR[k, -1, j] = Uk[g+Nx]

        if swap:
            UL = UL.swapaxes(1, 2)
            UR = UR.swapaxes(1, 2)
        return UL, UR

    raise ValueError("axis debe ser None (1-D), 0 (x) o 1 (y)")
 """

"""
Reconstrucción 1-D / 2-D para esquemas FV con celdas fantasma.

Convención de índices (1-D)
---------------------------
          g-2  g-1 |  g   …  g+Nx-1 | g+Nx g+Nx+1
caras:      ^   0  ^  1  …   Nx-1  ^  Nx   ^

•  Las celdas g … g+Nx-1 son físicas.
•  UL/UR[0] provienen de (g-1, g)      → interface x = xmin
•  UL/UR[Nx] de (g+Nx-1, g+Nx)         → interface x = xmax
Con BC periódicas, las ghost ya contienen la copia correcta,
de modo que UL/UR[0] ≃ UL/UR[Nx] (flujo continuo).
"""
import numpy as np
from config import NGHOST


# ------------------------------------------------------------
# 1.  Utilidades y limitadores básicos
# ------------------------------------------------------------
def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod3(a, b, c):
    pos = (a > 0) & (b > 0) & (c > 0)
    neg = (a < 0) & (b < 0) & (c < 0)
    out = np.where(pos, np.minimum(np.minimum(a, b), c), 0.0)
    out = np.where(neg, np.maximum(np.maximum(a, b), c), out)
    return out

def maxmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) > np.abs(b), a, b), 0.0)


# ------------------------------------------------------------
# 2.  Pendiente MUSCL (menos disipativa)
# ------------------------------------------------------------
def slope_full(U_phys, dx, limiter="minmod"):
    """
    Pendiente centrada para todas las celdas físicas.
    Devuelve vector de longitud Nx (mesma shape que U_phys).
    """
    if limiter == "minmod":
        a = (U_phys[1:] - U_phys[:-1]) / dx            # fwd
        b = (U_phys[:-1] - U_phys[1:]) / (-dx)         # back
        slope_inner = minmod(a[:-1], b[1:])
    elif limiter == "mc":
        a = (U_phys[2:] - U_phys[:-2]) / (2*dx)
        b = 2*(U_phys[1:-1] - U_phys[:-2]) / dx
        c = 2*(U_phys[2:]   - U_phys[1:-1]) / dx
        slope_inner = minmod3(a, b, c)
    elif limiter == "superbee":
        s1 = minmod(2*(U_phys[1:-1]-U_phys[:-2])/dx,
                    (U_phys[2:]-U_phys[1:-1])/dx)
        s2 = minmod((U_phys[1:-1]-U_phys[:-2])/dx,
                    2*(U_phys[2:]-U_phys[1:-1])/dx)
        slope_inner = maxmod(s1, s2)
    else:
        raise ValueError(f"Limitador '{limiter}' no reconocido")

    Nx = len(U_phys)
    slope = np.empty_like(U_phys)
    slope[1:-1] = slope_inner
    slope[0]    = slope_inner[0]       # copias en extremos
    slope[-1]   = slope_inner[-1]
    return slope


# ------------------------------------------------------------
# 3.  MP5 – caras internas (i+½)   (requiere NGHOST ≥ 2)
# ------------------------------------------------------------
def minmod_pair(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod4(w, x, y, z):
    s = 0.125*(np.sign(w)+np.sign(x))*np.abs((np.sign(w)+np.sign(y))*(np.sign(w)+np.sign(z)))
    return s*np.min(np.stack([np.abs(w), np.abs(x), np.abs(y), np.abs(z)]), axis=0)

def mp5_faces(U, dx, alpha=4.0, eps=1e-6):
    """
    Devuelve (vl, vr) para las Nx-1 caras internas 1…Nx-1
    suponiendo que `U` incluye al menos 2 ghost a cada lado.
    """
    if NGHOST < 2:
        raise ValueError("MP5 requiere NGHOST ≥ 2")

    um2, um1, u0, up1, up2 = U[:-4], U[1:-3], U[2:-2], U[3:-1], U[4:]
    vnorm  = np.sqrt(um2**2+um1**2+u0**2+up1**2+up2**2)

    # --- (i+½)^-
    vl   = (2*um2 - 13*um1 + 47*u0 + 27*up1 - 3*up2)/60
    vmp  = u0 + minmod_pair(up1-u0, alpha*(u0-um1))
    cond = (vl-u0)*(vl-vmp) <= eps*vnorm
    djm1 = um2 - 2*um1 + u0
    dj   = um1 - 2*u0 + up1
    djp1 = u0  - 2*up1 + up2
    dm4jmh = minmod4(4*dj-djm1, 4*djm1-dj, dj, djm1)
    dm4jph = minmod4(4*dj-djp1, 4*djp1-dj, dj, djp1)
    vul = u0 + alpha*(u0-um1)
    vav = 0.5*(u0+up1)
    vmd = vav - 0.5*dm4jph
    vlc = u0 + 0.5*(u0-um1) + (4/3)*dm4jmh
    vminl = np.maximum(np.minimum.reduce([u0, up1, vmd]),
                       np.minimum.reduce([u0, vul, vlc]))
    vmaxl = np.minimum(np.maximum.reduce([u0, up1, vmd]),
                       np.maximum.reduce([u0, vul, vlc]))
    vl += minmod_pair(vminl-vl, vmaxl-vl)
    vl  = np.where(cond, vl, vl)

    # --- (i-½)^+
    vr   = (2*up2 - 13*up1 + 47*u0 + 27*um1 - 3*um2)/60
    vmp  = u0 + minmod_pair(um1-u0, alpha*(u0-up1))
    cond = (vr-u0)*(vr-vmp) <= eps*vnorm
    djm1 = up2 - 2*up1 + u0
    dj   = up1 - 2*u0 + um1
    djp1 = u0  - 2*um1 + um2
    dm4jmh = minmod4(4*dj-djm1, 4*djm1-dj, dj, djm1)
    dm4jph = minmod4(4*dj-djp1, 4*djp1-dj, dj, djp1)
    vul = u0 + alpha*(u0-up1)
    vav = 0.5*(u0+um1)
    vmd = vav - 0.5*dm4jph
    vlc = u0 + 0.5*(u0-up1) + (4/3)*dm4jmh
    vminr = np.maximum(np.minimum.reduce([u0, um1, vmd]),
                       np.minimum.reduce([u0, vul, vlc]))
    vmaxr = np.minimum(np.maximum.reduce([u0, um1, vmd]),
                       np.maximum.reduce([u0, vul, vlc]))
    vr += minmod_pair(vminr-vr, vmaxr-vr)
    vr  = np.where(cond, vr, vr)

    return vl, vr   # ambas de longitud Nx-1


# ------------------------------------------------------------
# 4.  Reconstrucción principal  (1-D y 2-D)
# ------------------------------------------------------------
def reconstruct(U, dx, *, limiter="minmod", axis=None):
    """
    Devuelve UL, UR en todas las interfaces del dominio físico:

      • axis=None → 1-D
      • axis=0    → interfaces normales a x  (2-D)
      • axis=1    → interfaces normales a y  (2-D)
    """
    g = NGHOST

    # ---------- 1-D ----------
    if axis is None:
        nvars, Ntot = U.shape
        Nx = Ntot - 2*g
        UL = np.empty((nvars, Nx+1))
        UR = np.empty((nvars, Nx+1))

        for k in range(nvars):
            Uk = U[k]
            if limiter == "mp5":
                vl, vr = mp5_faces(Uk, dx)           # len = Nx+2g-4
                start = g - 2                       # g = NGHOST
                stop  = start + (Nx - 1)            # Nx-1 caras que queremos
                vl = vl[start:stop]
                vr = vr[start:stop]
                UL[k, 1:Nx]  = vl                   # 1 … Nx-1
                UR[k, 1:Nx]  = vr
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:Nx]  = Uk[g:-g] + 0.5*dx*slope
                UR[k, 1:Nx]  = Uk[g:-g] - 0.5*dx*slope

            # Interfaces externas (usan fantasma)
            UL[k, 0]   = Uk[g-1]
            UR[k, 0]   = Uk[g]
            UL[k, -1]  = Uk[g+Nx-1]
            UR[k, -1]  = Uk[g+Nx]

        return UL, UR

    # ---------- 2-D ----------
    if axis not in (0, 1):
        raise ValueError("axis debe ser None, 0 (x) o 1 (y)")

    swap = (axis == 1)
    if swap:
        U = U.swapaxes(1, 2)   # trabajamos siempre a lo largo de x

    nvars, Nxtot, Ny = U.shape
    Nx = Nxtot - 2*g
    UL = np.empty((nvars, Nx+1, Ny))
    UR = np.empty((nvars, Nx+1, Ny))

    for k in range(nvars):
        for j in range(Ny):
            Uk = U[k, :, j]

            if limiter == "mp5":
                vl, vr = mp5_faces(Uk, dx)
                UL[k, 1:Nx, j]  = vl
                UR[k, 1:Nx, j]  = vr
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:Nx, j]  = Uk[g:-g] + 0.5*dx*slope
                UR[k, 1:Nx, j]  = Uk[g:-g] - 0.5*dx*slope

            UL[k, 0,  j] = Uk[g-1]       # izquierda
            UR[k, 0,  j] = Uk[g]
            UL[k, -1, j] = Uk[g+Nx-1]    # derecha
            UR[k, -1, j] = Uk[g+Nx]

    if swap:        # volver a (nvars, Nx, Ny+1) orden original
        UL = UL.swapaxes(1, 2)
        UR = UR.swapaxes(1, 2)
    return UL, UR
