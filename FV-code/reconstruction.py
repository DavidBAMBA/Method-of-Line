# reconstruct.py
import numpy as np
# ============================================================
#  Limitadores básicos
# ============================================================
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

# --- utilidades MP5 ---
def minmod_pair(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod4(w, x, y, z):
    s = 0.125 * (np.sign(w) + np.sign(x)) * np.abs((np.sign(w) + np.sign(y)) *
                                                   (np.sign(w) + np.sign(z)))
    return s * np.min(np.stack([np.abs(w), np.abs(x), np.abs(y), np.abs(z)]), axis=0)

# ============================================================
#  Slopes MUSCL (interiores Nx-2)
# ============================================================
def _slope_minmod_inner(U, dx):
    return minmod((U[1:-1] - U[:-2]) / dx, (U[2:] - U[1:-1]) / dx)

def _slope_mc_inner(U, dx):
    a = (U[2:] - U[:-2]) / (2*dx)
    b = 2 * (U[1:-1] - U[:-2]) / dx
    c = 2 * (U[2:]   - U[1:-1]) / dx
    return minmod3(a, b, c)

def _slope_superbee_inner(U, dx):
    s1 = minmod(2*(U[1:-1] - U[:-2]) / dx, (U[2:] - U[1:-1]) / dx)
    s2 = minmod((U[1:-1] - U[:-2]) / dx, 2*(U[2:] - U[1:-1]) / dx)
    return maxmod(s1, s2)

def slope_full(U, dx, kind):
    """
    Devuelve un vector de longitud Nx con la pendiente MUSCL.
    Extiende los extremos copiando la primera/última pendiente interior.
    """
    if kind == "minmod":
        s_inner = _slope_minmod_inner(U, dx)
    elif kind == "mc":
        s_inner = _slope_mc_inner(U, dx)
    elif kind == "superbee":
        s_inner = _slope_superbee_inner(U, dx)
    else:
        raise ValueError

    slope = np.empty_like(U)
    slope[1:-1] = s_inner
    slope[0]  = s_inner[0]
    slope[-1] = s_inner[-1]
    return slope

# ============================================================
#  MP5: devuelve directamente las caras internas
# ============================================================
def mp5_faces(U, dx, alpha=4.0, eps=1.0e-6):
    um2, um1, u0, up1, up2 = U[:-4], U[1:-3], U[2:-2], U[3:-1], U[4:]
    vnorm = np.sqrt(um2**2 + um1**2 + u0**2 + up1**2 + up2**2)
    vl = (2*um2 - 13*um1 + 47*u0 + 27*up1 - 3*up2) / 60.0
    vmp = u0 + minmod_pair(up1 - u0, alpha*(u0 - um1))
    smooth = (vl - u0)*(vl - vmp) <= eps*vnorm
    # --- corrección ---
    djm1 = um2 - 2*um1 + u0
    dj   = um1 - 2*u0 + up1
    djp1 = u0  - 2*up1 + up2
    dm4jph = minmod4(4*dj - djp1, 4*djp1 - dj, dj, djp1)
    dm4jmh = minmod4(4*dj - djm1, 4*djm1 - dj, dj, djm1)
    vul = u0 + alpha*(u0 - um1)
    vav = 0.5*(u0 + up1)
    vmd = vav - 0.5*dm4jph
    vlc = u0 + 0.5*(u0 - um1) + (4/3)*dm4jmh
    vmin = np.maximum(
                np.minimum.reduce([u0, up1, vmd]),
                np.minimum.reduce([u0, vul, vlc])
            )

    vmax = np.minimum(
                np.maximum.reduce([u0, up1, vmd]),
                np.maximum.reduce([u0, vul, vlc])
            )
    vl_corr = vl + minmod_pair(vmin - vl, vmax - vl)
    vl = np.where(smooth, vl, vl_corr)
    vr = np.roll(vl, -1)
    return vl[:-1], vr[:-1]          # Nx-1 interfaces internas

# ============================================================
#  Reconstrucción UL/UR
# ============================================================
def reconstruct(U, dx, limiter="minmod", axis=None):
    """
    UL, UR en todas las interfaces (Nx+1 1-D).
    """
    # ---------------- 1-D ----------------
    if axis is None:
        nvars, Nx = U.shape
        UL = np.zeros((nvars, Nx+1))
        UR = np.zeros((nvars, Nx+1))

        for k in range(nvars):
            Uk = U[k]

            if limiter == "mp5":
                Uext = np.pad(Uk, (2,2), mode='edge')
                vl, vr = mp5_faces(Uext, dx)
                UL[k,1:-1] = vl
                UR[k,1:-1] = vr
            else:
                slope = slope_full(Uk, dx, limiter)
                UL[k,1:]  = Uk + 0.5*dx*slope
                UR[k,:-1] = Uk - 0.5*dx*slope

            # extremos (1st-order)
            UL[k,0]  = Uk[0];  UR[k,0]  = Uk[0]
            UL[k,-1] = Uk[-1]; UR[k,-1] = Uk[-1]

        return UL, UR

    # ---------------- 2-D eje x ----------------
    if axis == 0:
        nvars, Nx, Ny = U.shape
        UL = np.zeros((nvars, Nx+1, Ny))
        UR = np.zeros((nvars, Nx+1, Ny))

        for k in range(nvars):
            for j in range(Ny):
                Uk = U[k,:,j]
                if limiter == "mp5":
                    Uext = np.pad(Uk, (2,2), mode='edge')
                    vl, vr = mp5_faces(Uext, dx)
                    UL[k,1:-1,j] = vl
                    UR[k,1:-1,j] = vr
                else:
                    slope = slope_full(Uk, dx, limiter)
                    UL[k,1:,j]  = Uk + 0.5*dx*slope
                    UR[k,:-1,j] = Uk - 0.5*dx*slope
                UL[k,0 ,j] = Uk[0];  UR[k,0 ,j] = Uk[0]
                UL[k,-1,j] = Uk[-1]; UR[k,-1,j] = Uk[-1]
        return UL, UR

    # ---------------- 2-D eje y ----------------
    if axis == 1:
        nvars, Nx, Ny = U.shape
        UL = np.zeros((nvars, Nx, Ny+1))
        UR = np.zeros((nvars, Nx, Ny+1))

        for k in range(nvars):
            for i in range(Nx):
                Uk = U[k,i,:]
                if limiter == "mp5":
                    Uext = np.pad(Uk, (2,2), mode='edge')
                    vl, vr = mp5_faces(Uext, dx)
                    UL[k,i,1:-1] = vl
                    UR[k,i,1:-1] = vr
                else:
                    slope = slope_full(Uk, dx, limiter)
                    UL[k,i,1:]  = Uk + 0.5*dx*slope
                    UR[k,i,:-1] = Uk - 0.5*dx*slope
                UL[k,i,0 ] = Uk[0];  UR[k,i,0 ] = Uk[0]
                UL[k,i,-1] = Uk[-1]; UR[k,i,-1] = Uk[-1]
        return UL, UR

    raise ValueError("axis debe ser None (1-D), 0 (x) o 1 (y)")
