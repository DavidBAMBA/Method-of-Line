import numpy as np
from config import NGHOST


# ─────────────────────────────────────────────────────────────
# 1. Utilidades MUSCL
# ─────────────────────────────────────────────────────────────
def minmod(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod3(a, b, c):
    pos = (a > 0) & (b > 0) & (c > 0)
    neg = (a < 0) & (b < 0) & (c < 0)
    out = np.where(pos, np.minimum(np.minimum(a, b), c), 0.0)
    out = np.where(neg, np.maximum(np.maximum(a, b), c), out)
    return out

def maxmod(a, b):
    return np.where(a*b > 0, np.where(np.abs(a) > np.abs(b), a, b), 0.0)

def slope_full(U_phys, dx, limiter="minmod"):
    if limiter == "minmod":
        a = (U_phys[1:] - U_phys[:-1]) / dx
        b = (U_phys[:-1] - U_phys[1:]) / (-dx)
        s = minmod(a[:-1], b[1:])
    elif limiter == "mc":
        a = (U_phys[2:] - U_phys[:-2]) / (2*dx)
        b = 2*(U_phys[1:-1] - U_phys[:-2]) / dx
        c = 2*(U_phys[2:]   - U_phys[1:-1]) / dx
        s = minmod3(a, b, c)
    elif limiter == "superbee":
        s1 = minmod(2*(U_phys[1:-1]-U_phys[:-2])/dx,
                    (U_phys[2:]-U_phys[1:-1])/dx)
        s2 = minmod((U_phys[1:-1]-U_phys[:-2])/dx,
                    2*(U_phys[2:]-U_phys[1:-1])/dx)
        s = maxmod(s1, s2)
    else:
        raise ValueError(f"Limitador '{limiter}' no reconocido")

    out = np.empty_like(U_phys)
    out[1:-1] = s
    out[0]  = s[0]
    out[-1] = s[-1]
    return out

# ─────────────────────────────────────────────────────────────
# 2. MP5  
# ─────────────────────────────────────────────────────────────
def minmod_pair(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod4(w, x, y, z):
    s = 0.125*(np.sign(w)+np.sign(x))*np.abs((np.sign(w)+np.sign(y))*(np.sign(w)+np.sign(z)))
    return s*np.min(np.stack([np.abs(w), np.abs(x), np.abs(y), np.abs(z)]), axis=0)

def mp5_faces(U, alpha=4.0, eps=1e-6):

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

# ─────────────────────────────────────────────────────────────
#   WENO-3           
# ─────────────────────────────────────────────────────────────
def _weno3_left(am1, a0, ap1):
    """Reconstrucción desde la izquierda hacia i+1/2"""
    beta0 = (ap1 - a0)**2
    beta1 = (a0  - am1)**2
    eps = 1e-12
    alpha0 = (2/3) / (eps + beta0)**2
    alpha1 = (1/3) / (eps + beta1)**2
    w0 = alpha0 / (alpha0 + alpha1)
    w1 = 1.0 - w0
    u0 = 0.5*a0 + 0.5*ap1
    u1 = -0.5*am1 + 1.5*a0
    return w0*u0 + w1*u1

def _weno3_right(a0, ap1, ap2):
    """Reconstrucción desde la derecha hacia i-1/2"""
    beta0 = (ap1 - ap2)**2
    beta1 = (a0  - ap1)**2
    eps = 1e-12
    alpha0 = (1/3) / (eps + beta0)**2
    alpha1 = (2/3) / (eps + beta1)**2
    w0 = alpha0 / (alpha0 + alpha1)
    w1 = 1.0 - w0
    u0 = 1.5*ap1 - 0.5*ap2
    u1 = 0.5*a0 + 0.5*ap1
    return w0*u0 + w1*u1

def weno3_faces(U):
    if NGHOST < 1:
        raise ValueError("WENO3 requiere ≥1 ghost cell")

    # Izquierda de i+1/2 con stencil i-1, i, i+1
    vl = _weno3_left(U[:-2], U[1:-1], U[2:])

    # Derecha de i-1/2 con stencil i, i+1, i+2
    vr = _weno3_right(U[:-2], U[1:-1], U[2:])

    return vl, vr


# ─────────────────────────────────────────────────────────────
#   WENO-5  clásico  (JS)           
# ─────────────────────────────────────────────────────────────
def _weno5_left(u):
    im2, im1, i, ip1, ip2 = u[:-4], u[1:-3], u[2:-2], u[3:-1], u[4:]
    beta0 = 13/12*(im2 - 2*im1 + i)**2   + 1/4*(im2 - 4*im1 + 3*i)**2
    beta1 = 13/12*(im1 - 2*i   + ip1)**2 + 1/4*(im1 - ip1)**2
    beta2 = 13/12*(i   - 2*ip1 + ip2)**2 + 1/4*(3*i - 4*ip1 + ip2)**2
    eps = 1.0e-6
    d0,d1,d2 = 0.1, 0.6, 0.3
    alpha0 = d0/(eps+beta0)**2
    alpha1 = d1/(eps+beta1)**2
    alpha2 = d2/(eps+beta2)**2
    w0 = alpha0/(alpha0+alpha1+alpha2)
    w1 = alpha1/(alpha0+alpha1+alpha2)
    w2 = alpha2/(alpha0+alpha1+alpha2)
    p0 =  (2*im2 - 7*im1 + 11*i)/6
    p1 = (-im1   + 5*i   + 2*ip1)/6
    p2 = (2*i    + 5*ip1 - ip2   )/6
    return w0*p0 + w1*p1 + w2*p2

def _weno5_right(u):
    return _weno5_left(u[::-1])[::-1]   # simetría

def weno5_faces(U):
    if NGHOST < 2:
        raise ValueError("WENO-5 requiere NGHOST ≥ 2")
    vl = _weno5_left(U)
    vr = _weno5_right(U)
    return vl, vr                   # longitud Ntot-4


#      WENO-Z  (Borges 2008)      
# ─────────────────────────────────────────────────────────────
def _wenoz_left(u):
    im2, im1, i, ip1, ip2 = u[:-4], u[1:-3], u[2:-2], u[3:-1], u[4:]
    beta0 = 13/12*(im2 - 2*im1 + i)**2   + 1/4*(im2 - 4*im1 + 3*i)**2
    beta1 = 13/12*(im1 - 2*i   + ip1)**2 + 1/4*(im1 - ip1)**2
    beta2 = 13/12*(i   - 2*ip1 + ip2)**2 + 1/4*(3*i - 4*ip1 + ip2)**2
    tau5  = np.abs(beta0 - beta2)
    eps = 1.0e-6
    d0,d1,d2 = 0.1, 0.6, 0.3
    alpha0 = d0*(1 + (tau5/(beta0+eps))**2)
    alpha1 = d1*(1 + (tau5/(beta1+eps))**2)
    alpha2 = d2*(1 + (tau5/(beta2+eps))**2)
    w0 = alpha0/(alpha0+alpha1+alpha2)
    w1 = alpha1/(alpha0+alpha1+alpha2)
    w2 = alpha2/(alpha0+alpha1+alpha2)
    p0 =  (2*im2 - 7*im1 + 11*i)/6
    p1 = (-im1   + 5*i   + 2*ip1)/6
    p2 = (2*i    + 5*ip1 - ip2   )/6
    return w0*p0 + w1*p1 + w2*p2

def _wenoz_right(u):
    return _wenoz_left(u[::-1])[::-1]

def wenoz_faces(U):
    if NGHOST < 2:
        raise ValueError("WENO-Z requiere NGHOST ≥ 2")
    vl = _wenoz_left(U)
    vr = _wenoz_right(U)
    return vl, vr

# ─────────────────────────────────────────────────────────────
# 4.  Función principal
# ─────────────────────────────────────────────────────────────
"""  
def reconstruct2(U, dx, *, limiter="minmod", axis=None):
    need = {"minmod":1, "mc":1, "superbee":1,
            "mp5":2, "weno3":1, "weno5":3, "wenoz":3}
    if NGHOST < need.get(limiter.lower(), 1):
        raise ValueError(f"{limiter.upper()} requiere NGHOST ≥ {need[limiter.lower()]}")

    g = NGHOST
    name = limiter.lower()

    # ============== 1-D =================================================
    if axis is None:
        nvars, Ntot = U.shape
        Nx   = Ntot - 2*g
        UL   = np.empty((nvars, Nx+1))
        UR   = np.empty((nvars, Nx+1))

        for k in range(nvars):
            Uk = U[k]

            # ---------- MP5 / WENO --------------------------------------
            if name in ("mp5", "weno5", "wenoz", "weno3"):
                if   name == "mp5":
                    vl, vr = mp5_faces(Uk);  off = 2
                elif name == "weno5":
                    vl, vr = weno5_faces(Uk); off = 2
                elif name == "wenoz":
                    vl, vr = wenoz_faces(Uk); off = 2
                else:   # weno3
                    vl, vr = weno3_faces(Uk);     off = 1

                # caras internas 1 … Nx-1
                start = g - off
                stop  = start + (Nx - 1)
                UL[k, 1:Nx] = vl[start:stop]
                UR[k, 1:Nx] = vr[start:stop]

                # caras periódicas 0 y Nx con el MISMO reconstructor
                UL[k, 0]  = vl[g - off - 1]        # = vl[g-off-1]
                UR[k, 0]  = vr[g - off]            # = vr[g-off]
                UL[k, -1] = vl[g + Nx - off - 1]   # = vl[g+Nx-off-1]
                UR[k, -1] = vr[g + Nx - off]       # = vr[g+Nx-off]

            # ---------- MUSCL lineales ----------------------------------
            else:     # minmod, mc, superbee
                # pendientes solo en celdas físicas 0 … Nx-1
                slope = slope_full(Uk[g:-g], dx, limiter)      # len = Nx

                # caras internas 1 … Nx-1
                UL[k, 1:]  = Uk[g:-g] + 0.5*dx*slope           # 1…Nx
                UR[k, :-1] = Uk[g:-g] - 0.5*dx*slope           # 0…Nx-1

                # caras periódicas 0 y Nx (con el mismo O(2))
                UL[k, 0]  = Uk[g+Nx-1] + 0.5*dx*slope[-1]      # celda Nx-1
                UR[k, 0]  = Uk[g]        - 0.5*dx*slope[0]     # celda 0
                UL[k, -1] = Uk[g+Nx-1] + 0.5*dx*slope[-1]      # misma celda
                UR[k, -1] = Uk[g]        - 0.5*dx*slope[0]

        return UL, UR

    # ============== 2-D =================================================
    if axis not in (0, 1):
        raise ValueError("axis debe ser None, 0 (x) o 1 (y)")

    swap = (axis == 1)            # si axis=1, trabajamos transpuesto en x
    if swap:
        U = U.swapaxes(1, 2)      # (nvars, Ny_tot , Nx_tot) → (nvars, Nx_tot, Ny)

    nvars, Nxtot, Ny = U.shape
    Nx = Nxtot - 2*g
    UL = np.empty((nvars, Nx+1, Ny))
    UR = np.empty((nvars, Nx+1, Ny))

    for k in range(nvars):
        for j in range(Ny):
            Uk = U[k, :, j]

            if name in ("mp5", "weno5", "wenoz", "weno3"):
                if   name == "mp5":
                    vl, vr = mp5_faces(Uk);  off = 2
                elif name == "weno5":
                    vl, vr = weno5_faces(Uk); off = 2
                elif name == "wenoz":
                    vl, vr = wenoz_faces(Uk); off = 2
                else:
                    vl, vr = weno3_faces(Uk); off = 1

                start = g - off
                stop  = start + (Nx - 1)
                UL[k, 1:Nx, j] = vl[start:stop]
                UR[k, 1:Nx, j] = vr[start:stop]

                UL[k, 0 , j]  = vl[g - off - 1]
                UR[k, 0 , j]  = vr[g - off]
                UL[k, -1, j]  = vl[g + Nx - off - 1]
                UR[k, -1, j]  = vr[g + Nx - off]

            else:   # minmod / mc / superbee
                slope = slope_full(Uk[g:-g], dx, limiter)      # len = Nx
                UL[k, 1:,  j] = Uk[g:-g] + 0.5*dx*slope
                UR[k, :-1, j] = Uk[g:-g] - 0.5*dx*slope

                UL[k, 0 , j] = Uk[g+Nx-1] + 0.5*dx*slope[-1]
                UR[k, 0 , j] = Uk[g]        - 0.5*dx*slope[0]
                UL[k, -1, j] = Uk[g+Nx-1] + 0.5*dx*slope[-1]
                UR[k, -1, j] = Uk[g]        - 0.5*dx*slope[0]

    if swap:                       # des-transponer si venía en y
        UL = UL.swapaxes(1, 2)
        UR = UR.swapaxes(1, 2)

    return UL, UR

# ------------------------------------------------------------------
def reconstruct1(U, dx, *, limiter="minmod", axis=None):
    g = NGHOST

    if axis is None:
        nvars, Ntot = U.shape
        Nx = Ntot - 2*g
        UL = np.zeros((nvars, Nx+1))
        UR = np.zeros((nvars, Nx+1))

        for k in range(nvars):
            Uk = U[k]
            if limiter == "mp5":
                vl, vr = mp5_faces(Uk, dx)         #  len(vl) = Nx-1   (caras 1 … Nx-1)
                UL[k, 1:]  = vl                 # caras   1 … Nx
                UR[k, :-1] = vr
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:]  = Uk[g:-g] + 0.5*dx*slope
                UR[k, :-1] = Uk[g:-g] - 0.5*dx*slope

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


def reconstruct(U, dx, *, limiter="minmod", axis=None, bc_x=None, bc_y=None):
    need = {"minmod": 1, "mc": 1, "superbee": 1,
            "mp5": 2, "weno3": 1, "weno5": 2, "wenoz": 2}
    if NGHOST < need.get(limiter.lower(), 1):
        raise ValueError(f"{limiter.upper()} requiere NGHOST ≥ {need[limiter.lower()]}")

    g = NGHOST
    name = limiter.lower()

    if axis is None or axis == 0:
        periodic = bc_x == ("periodic", "periodic")
    elif axis == 1:
        periodic = bc_y == ("periodic", "periodic")
    else:
        raise ValueError("axis debe ser None (1D), 0 (x) o 1 (y)")

    # ============== 1-D =================================================
    if axis is None:
        nvars, Ntot = U.shape
        Nx = Ntot - 2 * g
        UL = np.empty((nvars, Nx + 1))
        UR = np.empty((nvars, Nx + 1))

        for k in range(nvars):
            Uk = U[k]

            if name in ("mp5", "weno5", "wenoz", "weno3"):
                if name == "mp5":
                    vl, vr = mp5_faces(Uk)
                    off = 2
                elif name == "weno5":
                    vl, vr = weno5_faces(Uk)
                    off = 2
                elif name == "wenoz":
                    vl, vr = wenoz_faces(Uk)
                    off = 2
                else:
                    vl, vr = weno3_faces(Uk)
                    off = 1

                if periodic:
                    start = g - off
                    stop = start + (Nx - 1)
                    UL[k, 1:Nx] = vl[start:stop]
                    UR[k, 1:Nx] = vr[start:stop]
                    UL[k, 0] = vl[g - off - 1]
                    UR[k, 0] = vr[g - off]
                    UL[k, -1] = vl[g + Nx - off - 1]
                    UR[k, -1] = vr[g + Nx - off]
                else:
                    UL[k, 1:] = vl
                    UR[k, :-1] = vr
                    UL[k, 0] = Uk[g - 1]
                    UR[k, 0] = Uk[g]
                    UL[k, -1] = Uk[g + Nx - 1]
                    UR[k, -1] = Uk[g + Nx]
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:] = Uk[g:-g] + 0.5 * dx * slope
                UR[k, :-1] = Uk[g:-g] - 0.5 * dx * slope

                if periodic:
                    UL[k, 0] = Uk[g + Nx - 1] + 0.5 * dx * slope[-1]
                    UR[k, 0] = Uk[g] - 0.5 * dx * slope[0]
                    UL[k, -1] = Uk[g + Nx - 1] + 0.5 * dx * slope[-1]
                    UR[k, -1] = Uk[g] - 0.5 * dx * slope[0]
                else:
                    UL[k, 0] = Uk[g - 1]
                    UR[k, 0] = Uk[g]
                    UL[k, -1] = Uk[g + Nx - 1]
                    UR[k, -1] = Uk[g + Nx]

        return UL, UR

    # ============== 2-D =================================================
    swap = (axis == 1)
    if swap:
        U = U.swapaxes(1, 2)

    nvars, Nxtot, Ny = U.shape
    Nx = Nxtot - 2 * g
    UL = np.empty((nvars, Nx + 1, Ny))
    UR = np.empty((nvars, Nx + 1, Ny))

    for k in range(nvars):
        for j in range(Ny):
            Uk = U[k, :, j]

            if name in ("mp5", "weno5", "wenoz", "weno3"):
                if name == "mp5":
                    vl, vr = mp5_faces(Uk)
                    off = 2
                elif name == "weno5":
                    vl, vr = weno5_faces(Uk)
                    off = 2
                elif name == "wenoz":
                    vl, vr = wenoz_faces(Uk)
                    off = 2
                else:
                    vl, vr = weno3_faces(Uk)
                    off = 1

                if periodic:
                    start = g - off
                    stop = start + (Nx - 1)
                    UL[k, 1:Nx, j] = vl[start:stop]
                    UR[k, 1:Nx, j] = vr[start:stop]
                    UL[k, 0 , j] = vl[g - off - 1]
                    UR[k, 0 , j] = vr[g - off]
                    UL[k, -1, j] = vl[g + Nx - off - 1]
                    UR[k, -1, j] = vr[g + Nx - off]
                else:
                    UL[k, 1:, j] = vl
                    UR[k, :-1, j] = vr
                    UL[k, 0 , j] = Uk[g - 1]
                    UR[k, 0 , j] = Uk[g]
                    UL[k, -1, j] = Uk[g + Nx - 1]
                    UR[k, -1, j] = Uk[g + Nx]
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:, j] = Uk[g:-g] + 0.5 * dx * slope
                UR[k, :-1, j] = Uk[g:-g] - 0.5 * dx * slope

                if periodic:
                    UL[k, 0 , j] = Uk[g + Nx - 1] + 0.5 * dx * slope[-1]
                    UR[k, 0 , j] = Uk[g] - 0.5 * dx * slope[0]
                    UL[k, -1, j] = Uk[g + Nx - 1] + 0.5 * dx * slope[-1]
                    UR[k, -1, j] = Uk[g] - 0.5 * dx * slope[0]
                else:
                    UL[k, 0 , j] = Uk[g - 1]
                    UR[k, 0 , j] = Uk[g]
                    UL[k, -1, j] = Uk[g + Nx - 1]
                    UR[k, -1, j] = Uk[g + Nx]

    if swap:
        UL = UL.swapaxes(1, 2)
        UR = UR.swapaxes(1, 2)

    return UL, UR
 