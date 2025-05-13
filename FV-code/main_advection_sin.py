# main_advection_sin.py
"""
Advección 1-D con condición inicial de “critical points”
u0(x) = sin(πx − sin(πx)/π ), dominio [-1,1], BC periódicas.
Se compara la solución numérica (MP5) con la exacta.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re

# ── módulos del proyecto ─────────────────────────────────────────
from config        import NGHOST
from equations     import Advection1D
from initial_conditions import critical_sine_1d      # ← añádela si no existe
from solver        import RK4, dUdt
from reconstruction import reconstruct
from riemann        import solve_riemann
from utils          import create_mesh_1d, create_U0
from write          import setup_data_folder

# ─── parámetros ─────────────────────────────────────────────────
Nx        = 80
xmin, xmax = -1.0, 1.0
tf        = 8.0
cfl       = 0.1
velocity  = 1.0
limiter   = "mp5"
solver    = "exact"           # flujo upwind exacto para advección
prefix    = "adv_sin"

# ─── malla e IC ────────────────────────────────────────────────
x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
init       = critical_sine_1d(x_phys)
U0, _      = create_U0(nvars=1, shape_phys=(Nx,), initializer=init)

equation   = Advection1D(a=velocity)
recon      = lambda U,d,axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riem       = lambda UL,UR,eq,axis=None: solve_riemann(UL,UR,eq,axis, solver=solver)

# ─── integración ───────────────────────────────────────────────
setup_data_folder("data")
print(f"[INFO] Advección 1-D  • limiter={limiter.upper()} • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=None,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riem,
    x=x_phys, y=None,
    bc_x=("periodic","periodic"),
    cfl=cfl,
    save_every=10,        # sólo paso final
    filename=prefix,
    reconst=limiter)

# ─── leer CSV final ─────────────────────────────────────────────
csv = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"),
             key=lambda f: int(re.search(r"_(\d{5})\.csv$", f).group(1)))[-1]
data = np.loadtxt(csv, delimiter=",", skiprows=1)
u_num = data[:,1]
t_fin = data[0,2]

# ─── solución exacta en todo el rango de tiempos (para animar) ─
L = xmax - xmin
frames = 200
times  = np.linspace(0, tf, frames)

def exact(x,t):
    x_s = ((x - velocity*t - xmin) % L) + xmin
    return np.sin(np.pi*x_s - np.sin(np.pi*x_s)/np.pi)

u_exact_frames = np.array([exact(x_phys,t) for t in times])

# ─── animación Numérico vs Analítico ───────────────────────────
os.makedirs("videos", exist_ok=True)
fig, ax = plt.subplots(figsize=(10,4))
ln_num, = ax.plot(x_phys, u_exact_frames[0]*0, lw=2, label="Numérico")
ln_exa, = ax.plot(x_phys, u_exact_frames[0], "k--", label="Exacto")

ax.set(xlim=(xmin,xmax), ylim=(-1.2,1.2),
       xlabel="x", ylabel="u", title="Advección 1-D – evolución")
ax.legend()

def update(i):
    ti = times[i]
    # desplazando numérico (porque la ecuación es lineal)
    shift = int(round(ti/dx)) % Nx
    ln_num.set_ydata(np.roll(U0[0, NGHOST:-NGHOST], shift))
    ln_exa.set_ydata(u_exact_frames[i])
    ax.set_title(f"t = {ti:.2f}")
    return ln_num, ln_exa

ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
vpath = f"videos/{prefix}_{limiter}.mp4"
ani.save(vpath, writer=FFMpegWriter(fps=25, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación guardada en {vpath}")

# ─── figura final ──────────────────────────────────────────────
plt.figure(figsize=(10,4))
plt.plot(x_phys, u_num, "o", ms=3, label="Numérico")
plt.plot(x_phys, exact(x_phys,t_fin), "k--", lw=1.2, label="Exacto")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Estado final  t = {t_fin:.2f}")
plt.legend(); plt.tight_layout()
png = vpath.replace(".mp4",".png")
plt.savefig(png, dpi=300); plt.show()
print(f"[INFO] Figura final guardada en {png}")
