# main_burgers_1d.py
"""
Simulación de la ecuación de Burgers 1D con condiciones periódicas.
Perfil inicial gaussiano. Evaluación del reconstructor numérico.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re

# ── módulos del proyecto ─────────────────────────────────────────────────
from config             import NGHOST
from equations          import Burgers1D
from initial_conditions import gaussian_burgers_1d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0
from write              import setup_data_folder

# === Parámetros ===========================================================
Nx      = 400
xmin, xmax = 0.0, 1.0
tf      = 0.8
cfl     = 0.1
limiter = "mp5"
solver  = "hll"   # Riemann solver para Burgers
prefix  = "burgers_1d"

# === Malla y condición inicial ============================================
x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
init = gaussian_burgers_1d(x_phys, center=0.3, width=0.05, amp=1.0)
U0, phys = create_U0(nvars=1, shape_phys=(Nx,), initializer=init)

equation = Burgers1D()

# Wrappers -----------------------------------------------------------------
recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riem  = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=solver)

# === Simulación ===========================================================
setup_data_folder("data")
print(f"[INFO] Burgers 1-D  • limiter={limiter.upper()} • periodic • NGHOST={NGHOST}")

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

# === Leer CSVs ============================================================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pat = f"data/{prefix}_{limiter}_*.csv"
csv_files = sorted(
    glob.glob(pat),
    key=lambda f: int(re.search(r"_(\d{5})\.csv$", f).group(1)) if re.search(r"_(\d{5})\.csv$", f) else -1
)

# Filtrar archivos mal formateados
csv_files = [f for f in csv_files if re.search(r"_(\d{5})\.csv$", f)]

frames_u = []
times = []

for f in csv_files:
    try:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        u = data[:, 1]  # asumiendo columna 1 es u
        t = data[0, 2]  # asumiendo columna 2 es tiempo
        frames_u.append(u)
        times.append(t)
    except Exception as e:
        print(f"[WARN] No se pudo leer archivo {f}: {e}")

frames_u = np.asarray(frames_u)
times    = np.asarray(times)

if len(frames_u) == 0:
    raise RuntimeError("No se encontraron archivos válidos para graficar.")

# === Animación ============================================================
os.makedirs("videos", exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 4))

ln, = ax.plot(x_phys, frames_u[0], label="Numérico", color="tab:blue")
ax.set_xlim(xmin, xmax)
ax.set_ylim(frames_u.min()-0.1, frames_u.max()+0.1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Burgers 1D – evolución")
ax.legend()

def _update(i):
    ln.set_ydata(frames_u[i])
    ax.set_title(f"Burgers 1-D   t = {times[i]:.3f}")
    return ln,

video_path = f"videos/{prefix}_{limiter}.mp4"
ani = FuncAnimation(fig, _update, frames=len(times), interval=40, blit=True)
ani.save(video_path, writer=FFMpegWriter(fps=60, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación guardada en {video_path}")

# === Figura final =========================================================
u_num = frames_u[-1]

plt.figure(figsize=(10, 4))
plt.plot(x_phys, u_num, '-', lw=2, label="Numérico")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Burgers 1-D • {limiter.upper()} • t = {tf}")
plt.legend()
plt.tight_layout()

png_path = video_path.replace(".mp4", ".png")
plt.savefig(png_path, dpi=300)
plt.show()
print(f"[INFO] Figura final guardada en {png_path}")
