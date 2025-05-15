# main_advection_1d.py
"""
Prueba de advección escalar 1D con condiciones periódicas.
Comparación entre solución numérica y analítica (Gaussiana).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re

# ── módulos del proyecto ────────────────────────────────────────────────
from config             import NGHOST
from equations          import Advection1D
from initial_conditions import gaussian_advection_1d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0
from write              import setup_data_folder

# === Parámetros ===========================================================
Nx      = 200
xmin, xmax = 0.0, 1.0
tf      = 5.0
cfl     = 0.1
velocity = 1.0
limiter = "mp5"
solver = "hll"
prefix  = "advection_1d_hllc"

# === Malla y condición inicial ============================================
x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
init = gaussian_advection_1d(x_phys, center=0.5)
U0, phys = create_U0(nvars=1, shape_phys=(Nx,), initializer=init)

equation = Advection1D(a=velocity)

# Wrappers -----------------------------------------------------------------
recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riem  = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=solver)

# === Simulación ===========================================================
setup_data_folder("data")
print(f"[INFO] Advección 1-D  • limiter={limiter.upper()} • periodic • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=None,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riem,
    x=x_phys, y=None,
    bc_x=("periodic", "periodic"),
    cfl=cfl,
    save_every=10,
    filename=prefix,
    reconst=limiter)

# === Leer todos los CSV ===================================================
pat = f"data/{prefix}_{limiter}_*.csv"
csv_files = sorted(glob.glob(pat),
                   key=lambda f: int(re.search(r"_(\d{5})\.csv$", f).group(1)))

frames_u = []
times = []

for f in csv_files:
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    u = data[:,1]
    t = data[0,2]
    frames_u.append(u)
    times.append(t)

frames_u = np.asarray(frames_u)
times    = np.asarray(times)

# === Solución exacta para cada t ==========================================
L = xmax - xmin
analytical = []

for t in times:
    shift = (x_phys - velocity * t) % L
    ua = np.exp(-300 * (shift - 0.5)**2)
    analytical.append(ua)

analytical = np.asarray(analytical)

# === Animación Numérico vs Analítico ======================================
os.makedirs("videos", exist_ok=True)
fig, ax = plt.subplots(figsize=(10,4))

ln_num, = ax.plot(x_phys, frames_u[0], label="Num", color="tab:blue")
ln_exa, = ax.plot(x_phys, analytical[0], '--', label="Exacto", color="black")

ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Advección 1D – evolución")
ax.legend()

def _update(i):
    ln_num.set_ydata(frames_u[i])
    ln_exa.set_ydata(analytical[i])
    ax.set_title(f"Advección 1-D   t = {times[i]:.3f}")
    return ln_num, ln_exa

ani = FuncAnimation(fig, _update,
                    frames=len(times),
                    interval=40, blit=True)

video_path = f"videos/{prefix}_{limiter}.mp4"
ani.save(video_path, writer=FFMpegWriter(fps=60, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación guardada en {video_path}")

# === Figura final =========================================================
u_num = frames_u[-1]
u_exa = analytical[-1]

plt.figure(figsize=(10,4))
plt.plot(x_phys, u_num, 'o', ms=3, label="Num.")
plt.plot(x_phys, u_exa, '-', lw=2, label="Exacto")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Advección 1-D • {limiter.upper()} • t = {tf}")
plt.legend()
plt.tight_layout()

png_path = video_path.replace(".mp4", ".png")
plt.savefig(png_path, dpi=300)
plt.show()
print(f"[INFO] Figura final guardada en {png_path}")
