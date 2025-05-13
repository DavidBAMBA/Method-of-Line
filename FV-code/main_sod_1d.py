# main_sod_1d.py
"""
Prueba Sod shock tube con esquemas MUSCL/WENO y ghost-cells genéricos.
BC elegidas: **outflow** (gradiente cero) en ambos extremos.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re
from sod_exact import general_shock_tube_solution   # ←  FUNCIONES ANALÍTICAS

# -------------------------------------------------------------------------
from config             import NGHOST
from equations          import Euler1D
from initial_conditions import sod_shock_tube_1d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0
from write              import setup_data_folder

# === Parámetros ===========================================================
Nx   = 200
xmin, xmax = 0.0, 1.0
tf   = 0.2
cfl  = 0.1
gamma   = 1.4
limiter = "mp5"        # "minmod", "mc", "superbee"
riemann = "hllc"       # "hll" ó "hllc"
prefix  = "sod_1d"

# === Malla y condición inicial ============================================
x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
init = sod_shock_tube_1d(x_phys, x0=0.5)
U0, phys   = create_U0(nvars=3, shape_phys=(Nx,),
                       initializer=init)

equation = Euler1D(gamma=gamma)

# Wrappers -----------------------------------------------------------------
recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riem  = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis,
                                                    solver=riemann)

# === Simulación ===========================================================
setup_data_folder("data")
print(f"[INFO] Sod 1-D  • limiter={limiter.upper()} • solver={riemann.upper()} • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=None,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riem,
    x=x_phys, y=None,
    bc_x=("outflow", "outflow"),    # fronteras abiertas
    reflect_idx=[1],                # signo de ρv se invertiría si usas "reflect"
    cfl=cfl,
    save_every=100,
    filename=prefix,
    reconst=limiter)

# === Leer todos los CSV ===================================================
pat = f"data/{prefix}_{limiter}_*.csv"
csv_files = sorted(glob.glob(pat),
                   key=lambda f: int(re.search(r"_(\d{6})\.csv$", f).group(1)))

frames_rho, frames_v, frames_P, frames_e = [], [], [], []
times = []

for f in csv_files:
    dat = np.loadtxt(f, delimiter=",", skiprows=1)
    rho, v, P = dat[:,1], dat[:,2], dat[:,3]
    t         = dat[0,4]

    frames_rho.append(rho)
    frames_v.append(v)
    frames_P.append(P)
    frames_e.append(P / ((gamma-1.0)*rho))
    times.append(t)

frames_rho  = np.asarray(frames_rho)
frames_v    = np.asarray(frames_v)
frames_P    = np.asarray(frames_P)
frames_e    = np.asarray(frames_e)
times       = np.asarray(times)

# === Solución exacta para cada instante ===================================
rhoL,uL,pL = 1.0, 0.0, 1.0
rhoR,uR,pR = 0.125, 0.0, 0.1
analytic_rho, analytic_v, analytic_P, analytic_e = [], [], [], []

for t in times:
    rhoa, va, Pa, ea = general_shock_tube_solution(
        x_phys, t,
        rhoL, uL, pL,
        rhoR, uR, pR,
        gamma=gamma, x0=0.5
    )
    analytic_rho.append(rhoa)
    analytic_v.append(va)
    analytic_P.append(Pa)
    analytic_e.append(ea)

analytic_rho = np.asarray(analytic_rho)
analytic_v   = np.asarray(analytic_v)
analytic_P   = np.asarray(analytic_P)
analytic_e   = np.asarray(analytic_e)

# === Animación 1-D  (ρ, v, P  numérico vs. exacto) =======================
os.makedirs("videos", exist_ok=True)
fig, ax = plt.subplots(figsize=(10,5))

ln_rho_num, = ax.plot(x_phys, frames_rho[0], label='rho num', color='tab:blue')
ln_rho_exact, = ax.plot(x_phys, analytic_rho[0], '--', label='rho exact', color='k')

ln_v_num,  = ax.plot(x_phys, frames_v[0],  label='v num',  color='tab:orange')
ln_v_exact,  = ax.plot(x_phys, analytic_v[0],  '--', label='v exact',  color='gray')

ln_P_num,  = ax.plot(x_phys, frames_P[0],  label='P num',  color='tab:green')
ln_P_exact,  = ax.plot(x_phys, analytic_P[0],  '--', label='P exact',  color='lime')

ax.set(xlim=(xmin, xmax),
       ylim=(0, 1.1*max(frames_rho.max(), frames_P.max())),
       xlabel='x', ylabel='valor')
ax.legend(ncol=3, fontsize='small')

def _update(i):
    ln_rho_num.set_ydata(frames_rho[i])
    ln_rho_exact.set_ydata(analytic_rho[i])

    ln_v_num.set_ydata(frames_v[i])
    ln_v_exact.set_ydata(analytic_v[i])

    ln_P_num.set_ydata(frames_P[i])
    ln_P_exact.set_ydata(analytic_P[i])

    ax.set_title(f"Sod 1-D   t = {times[i]:.4f}")
    return (ln_rho_num, ln_rho_exact,
            ln_v_num,  ln_v_exact,
            ln_P_num,  ln_P_exact)

ani = FuncAnimation(fig, _update,
                    frames=len(times),
                    interval=40, blit=True)
out_mp4 = f"videos/{prefix}_{limiter}.mp4"
ani.save(out_mp4, writer=FFMpegWriter(fps=60, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación guardada en {out_mp4}")

# === Figura final (4 subplots) ============================================
x_an = x_phys   # para coherencia
rhoa , va , Pa , ea = analytic_rho[-1], analytic_v[-1], analytic_P[-1], analytic_e[-1]
rhon , vn , Pn , en = frames_rho[-1], frames_v[-1], frames_P[-1], frames_e[-1]

fig, axs = plt.subplots(2,2, figsize=(14,10), sharex=True)
# ρ
axs[0,0].plot(x_an, rhon, 'o', ms=3, label='Num.')
axs[0,0].plot(x_an, rhoa, '-', lw=2, label='Exacto')
axs[0,0].set_ylabel(r'$\rho$'); axs[0,0].legend()
# v
axs[0,1].plot(x_an, vn, 'o', ms=3)
axs[0,1].plot(x_an, va, '-', lw=2)
axs[0,1].set_ylabel('v')
# P
axs[1,0].plot(x_an, Pn, 'o', ms=3)
axs[1,0].plot(x_an, Pa, '-', lw=2)
axs[1,0].set_xlabel('x'); axs[1,0].set_ylabel('P')
# e_int
axs[1,1].plot(x_an, en, 'o', ms=3)
axs[1,1].plot(x_an, ea, '-', lw=2)
axs[1,1].set_xlabel('x'); axs[1,1].set_ylabel(r'$e_{\rm int}$')

fig.suptitle(f"Sod 1-D • {limiter.upper()} • t = {times[-1]:.4f}")
plt.tight_layout(rect=[0,0,1,0.96])
png_final = f"videos/{prefix}_{limiter}.png"
plt.savefig(png_final, dpi=300)
plt.close(fig)
print(f"[INFO] Figura final guardada en {png_final}")