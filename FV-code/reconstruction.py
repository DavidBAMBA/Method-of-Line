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



def _weno3_coeffs(a_m1, a_0, a_p1):
    beta0 = (a_p1 - a_0) ** 2
    beta1 = (a_0  - a_m1) ** 2
    eps   = 1.0e-12
    alpha0 = (2/3) / (eps + beta0)**2
    alpha1 = (1/3) / (eps + beta1)**2
    w0 = alpha0 / (alpha0 + alpha1)
    w1 = 1.0  - w0
    u0 =  0.5*a_0 + 0.5*a_p1          # est. polinomio 0
    u1 = -0.5*a_m1 + 1.5*a_0          # est. polinomio 1
    return w0*u0 + w1*u1

def _weno3_faces_1d(U):

    if NGHOST < 1:
        raise ValueError("WENO3 requiere ≥1 ghost cell")
    # índices convenientes
    a_m1 = U[:-2]   # i-1
    a_0  = U[1:-1]  # i
    a_p1 = U[2:]    # i+1
    # izquierda de i+½   con stencil i-1,i,i+1
    vl = _weno3_coeffs(a_m1, a_0, a_p1)
    # derecha de i-½  (equivalente, espejo)
    # para cara i-½ usamos celdas i-2,i-1,i
    vr = _weno3_coeffs(a_p1[::-1], a_0[::-1], a_m1[::-1])[::-1]
    return vl, vr

def weno3_faces(U, dx):
    return _weno3_faces_1d(U)



# ------------------------------------------------------------
# 3-C  WENO-5 (Jiang–Shu, k = 3)
#      requiere NGHOST ≥ 3  (usa Ui-2…Ui+2)
# ------------------------------------------------------------
def _js_weno5_left(U):
    # sténciles
    f_im2, f_im1, f_i, f_ip1, f_ip2 = U[:-4], U[1:-3], U[2:-2], U[3:-1], U[4:]
    # polinomios candidatos
    p0 = ( 2*f_im2 - 7*f_im1 + 11*f_i ) / 6
    p1 = (-f_im1 + 5*f_i + 2*f_ip1) / 6
    p2 = ( 2*f_i + 5*f_ip1 - f_ip2) / 6
    # indicadores de suavidad β
    b0 = (13/12)*(f_im2 - 2*f_im1 + f_i   )**2 + (1/4)*(f_im2 - 4*f_im1 + 3*f_i)**2
    b1 = (13/12)*(f_im1 - 2*f_i   + f_ip1 )**2 + (1/4)*(f_im1 - f_ip1          )**2
    b2 = (13/12)*(f_i    - 2*f_ip1 + f_ip2)**2 + (1/4)*(3*f_i  - 4*f_ip1 + f_ip2)**2
    eps = 1.0e-12
    a0 = 0.1 / (eps + b0)**2
    a1 = 0.6 / (eps + b1)**2
    a2 = 0.3 / (eps + b2)**2
    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = 1.0 - w0 - w1
    return w0*p0 + w1*p1 + w2*p2

def _js_weno5_right(U):
    # simetría → aplicar left al array invertido y luego invertir
    return _js_weno5_left(U[::-1])[::-1]

def weno5_faces(U, dx):

    if NGHOST < 3:
        raise ValueError("WENO-5 requiere NGHOST ≥ 3")
    vl = _js_weno5_left(U)     # caras 1 … Nx-2   respecto a dominio extendido
    vr = _js_weno5_right(U)
    return vl, vr



# ------------------------------------------------------------
# 4.  Reconstrucción principal  (1-D y 2-D)
# ------------------------------------------------------------
def reconstruct(U, dx, *, limiter="minmod", axis=None):

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
            elif limiter.lower() == "weno3":
                vl, vr = weno3_faces(Uk, dx)
                start, stop = g-1, g-1 + (Nx-1)
                UL[k,1:Nx], UR[k,1:Nx] = vl[start:stop], vr[start:stop]
            elif limiter.lower() == "weno5":
                # ---   caras 1 … Nx-1   --------------------------------------
                vl, vr = weno5_faces(Uk, dx)        # longitud Ntot-4
                start  = g - 2                      # <- ESTE es el offset correcto
                stop   = start + (Nx - 1)           # Nx-1 caras internas
                UL[k, 1:Nx] = vl[start:stop]
                UR[k, 1:Nx] = vr[start:stop]
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                #UL[k, 1:Nx]  = Uk[g:-g] + 0.5*dx*slope
                #UR[k, 1:Nx]  = Uk[g:-g] - 0.5*dx*slope
                UL[k, 1:]   = Uk[g:-g] + 0.5*dx*slope   # 1…Nx
                UR[k, :-1]  = Uk[g:-g] - 0.5*dx*slope 

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
        U = U.swapaxes(1, 2)   # trabajamos a lo largo de x

    nvars, Nxtot, Ny = U.shape
    Nx = Nxtot - 2*g
    UL = np.empty((nvars, Nx+1, Ny))
    UR = np.empty((nvars, Nx+1, Ny))

    for k in range(nvars):
        for j in range(Ny):
            Uk = U[k, :, j]

            if limiter == "mp5":
                vl, vr = mp5_faces(Uk, dx)  # largo = Nxtot - 4
                start = g - 2
                stop  = start + (Nx - 1)
                vl = vl[start:stop]
                vr = vr[start:stop]
                UL[k, 1:Nx, j] = vl
                UR[k, 1:Nx, j] = vr
            elif limiter.lower()=="weno3":
                vl,vr = weno3_faces(Uk,dx)
                start,stop = g-1, g-1+(Nx-1)
                UL[k,1:Nx,j], UR[k,1:Nx,j] = vl[start:stop], vr[start:stop]
            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:Nx, j] = Uk[g:-g] + 0.5 * dx * slope
                UR[k, 1:Nx, j] = Uk[g:-g] - 0.5 * dx * slope

            # bordes externos (izquierda y derecha)
            UL[k, 0,  j] = Uk[g - 1]
            UR[k, 0,  j] = Uk[g]
            UL[k, -1, j] = Uk[g + Nx - 1]
            UR[k, -1, j] = Uk[g + Nx]

    if swap:
        UL = UL.swapaxes(1, 2)
        UR = UR.swapaxes(1, 2)

    return UL, UR


 """


"""
Reconstrucción 1-D / 2-D para FV con celdas fantasma.
Añadido:  • WENO-Z (Borges et al. 2008, JCP 227, 3191)
          • WENO-5 clásico (JS)            (ya lo tenías)

NGHOST debe ser ≥ 3 para weno5 / wenoZ  (esténcil 5-puntos).
"""
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
# 2. MP5  (sin cambios)
# ─────────────────────────────────────────────────────────────
def minmod_pair(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)

def minmod4(w, x, y, z):
    s = 0.125*(np.sign(w)+np.sign(x))*np.abs((np.sign(w)+np.sign(y))*(np.sign(w)+np.sign(z)))
    return s*np.min(np.stack([np.abs(w), np.abs(x), np.abs(y), np.abs(z)]), axis=0)

def mp5_faces(U, dx, alpha=4.0, eps=1e-6):

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
#      WENO-Z  (Borges 2008)      
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

def weno5_faces(U, dx):
    if NGHOST < 3:
        raise ValueError("WENO-5 requiere NGHOST ≥ 3")
    vl = _weno5_left(U)
    vr = _weno5_right(U)
    return vl, vr                   # longitud Ntot-4


#      WENO-Z  (Borges 2008)      
# ─────────────────────────────────────────────────────────────
def _wenoZ_left(u):
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

def _wenoZ_right(u):
    return _wenoZ_left(u[::-1])[::-1]

def wenoZ_faces(U, dx):
    if NGHOST < 3:
        raise ValueError("WENO-Z requiere NGHOST ≥ 3")
    vl = _wenoZ_left(U)
    vr = _wenoZ_right(U)
    return vl, vr

# ─────────────────────────────────────────────────────────────
# 4.  Función principal
# ─────────────────────────────────────────────────────────────
def reconstruct(U, dx, *, limiter="minmod", axis=None):
    """
    limiter ∈ {"minmod","mc","superbee","mp5",
               "weno5","wenoZ"}
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

            if limiter.lower() == "mp5":
                vl, vr = mp5_faces(Uk, dx)
                start = g-2;  stop = start + (Nx-1)
                UL[k, 1:Nx] = vl[start:stop]
                UR[k, 1:Nx] = vr[start:stop]
            elif limiter.lower() == "weno3":
                vl, vr = weno3_faces(Uk)
                start, stop = g-1, g-1 + (Nx-1)
                UL[k,1:Nx] = vl[start:stop]
                UR[k,1:Nx] = vr[start:stop]

            elif limiter.lower() == "weno5":
                vl, vr = weno5_faces(Uk, dx)
                start = g-2;  stop = start + (Nx-1)
                UL[k, 1:Nx] = vl[start:stop]
                UR[k, 1:Nx] = vr[start:stop]

            elif limiter.lower() == "wenoz":
                vl, vr = wenoZ_faces(Uk, dx)
                start = g-2;  stop = start + (Nx-1)
                UL[k, 1:Nx] = vl[start:stop]
                UR[k, 1:Nx] = vr[start:stop]

            else: 
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k, 1:]  = Uk[g:-g] + 0.5*dx*slope
                UR[k, :-1] = Uk[g:-g] - 0.5*dx*slope

            # interfaces externas (usan ghost ya periódicas/reflejo/etc.)
            UL[k, 0]  = Uk[g-1]
            UR[k, 0]  = Uk[g]
            UL[k, -1] = Uk[g+Nx-1]
            UR[k, -1] = Uk[g+Nx]

        return UL, UR

    # ---------- 2-D ----------
    if axis not in (0,1):
        raise ValueError("axis debe ser None, 0 (x) o 1 (y)")

    swap = (axis == 1)
    if swap:  # trabajamos sobre x
        U = U.swapaxes(1,2)

    nvars, Nxtot, Ny = U.shape
    Nx = Nxtot - 2*g
    UL = np.empty((nvars, Nx+1, Ny))
    UR = np.empty((nvars, Nx+1, Ny))

    for k in range(nvars):
        for j in range(Ny):
            Uk = U[k,:,j]

            if limiter.lower() == "mp5":
                vl, vr = mp5_faces(Uk, dx)
                start = g-2;  stop = start + (Nx-1)
                UL[k,1:Nx,j] = vl[start:stop]
                UR[k,1:Nx,j] = vr[start:stop]

            elif limiter.lower() == "weno5":
                vl, vr = weno5_faces(Uk, dx)
                start = g-2;  stop = start + (Nx-1)
                UL[k,1:Nx,j] = vl[start:stop]
                UR[k,1:Nx,j] = vr[start:stop]

            elif limiter.lower() == "wenoz":
                vl, vr = wenoZ_faces(Uk, dx)
                start = g-2;  stop = start + (Nx-1)
                UL[k,1:Nx,j] = vl[start:stop]
                UR[k,1:Nx,j] = vr[start:stop]

            else:
                slope = slope_full(Uk[g:-g], dx, limiter)
                UL[k,1:Nx,j] = Uk[g:-g] + 0.5*dx*slope
                UR[k,1:Nx,j] = Uk[g:-g] - 0.5*dx*slope

            UL[k,0 ,j] = Uk[g-1]
            UR[k,0 ,j] = Uk[g]
            UL[k,-1,j] = Uk[g+Nx-1]
            UR[k,-1,j] = Uk[g+Nx]

    if swap:
        UL = UL.swapaxes(1,2)
        UR = UR.swapaxes(1,2)
    return UL, UR
