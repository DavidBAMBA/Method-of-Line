# main_sod_2d.py  ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os, glob, re

# --- módulos del proyecto -------------------------------------------------
from config             import NGHOST
from equations          import Euler2D
from initial_conditions import sod_shock_tube_2d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_2d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
Nx, Ny        = 300, 150
xmin, xmax    = 0.0, 1.0
ymin, ymax    = 0.0, 1.0
tf            = 0.2
cfl           = 0.4
limiter       = "mp5"
riemann_name  = "hllc"
gamma         = 1.4
prefix        = "sod2d"

# ═══════════════════════════════ malla + IC ═══════════════════════════════
x_phys, y_phys, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
initializer            = sod_shock_tube_2d(x_phys, y_phys, x0=0.5)
U0, phys               = create_U0(nvars=4, shape_phys=(Nx, Ny), initializer=initializer)
equation               = Euler2D(gamma=gamma)

recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis,
                                            bc_x="outflow", bc_y="periodic")

riemann = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=riemann_name)

# ═════════════════════════════ simulación ════════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

print(f"[INFO] Sod 2-D • limiter={limiter.upper()} • solver={riemann_name.upper()} • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=dy,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riemann,
    x=x_phys, y=y_phys,
    bc_x=("outflow", "outflow"),
    bc_y=("periodic", "periodic"),
    cfl=cfl,
    save_every=10,
    filename=prefix,
    reconst=limiter)

# ═══════════════════════════  carga de CSVs  ══════════════════════════════
def _step(fname):
    return int(re.search(r"_(\d{5})\.csv$", fname).group(1))

csv_files = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"), key=_step)

frames_rho, times = [], []
for f in csv_files:
    data   = np.loadtxt(f, delimiter=",", skiprows=1)
    t_real = data[0, -1]
    rho    = data[:, 2].reshape((Nx, Ny))  # columnas: x, y, rho, v, P, time
    frames_rho.append(rho)
    times.append(t_real)

frames_rho = np.asarray(frames_rho)
times      = np.array(times)
# ══════════════════════  estado final 1-D (y = 0.5)  ═════════════════════
j = np.argmin(np.abs(y_phys - 0.5))
rho_slice = frames_rho[-1][:, j]

plt.figure(figsize=(6, 4))
plt.plot(x_phys, rho_slice, color='tab:blue')
plt.xlabel("x"); plt.ylabel("ρ(x, y=0.5)")
plt.title("Corte de densidad final en y = 0.5")
plt.xlim(xmin, xmax); plt.ylim(0, rho_slice.max()*1.1)
plt.tight_layout()
png_slice = f"videos/{prefix}_slice_y05_final.png"
plt.savefig(png_slice, dpi=300); plt.close()
print(f"[INFO] Corte final 1-D guardado en {png_slice}")

# ══════════════════════  animación 1-D de la evolución  ══════════════════
fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot([], [], color='tab:red')
ax.set_xlim(xmin, xmax)
ax.set_ylim(0, frames_rho.max()*1.05)
ax.set_xlabel("x")
ax.set_ylabel("ρ(x, y=0.5)")
ax.set_title("Evolución de ρ en y = 0.5")

def _update_1d(k):
    line.set_data(x_phys, frames_rho[k][:, j])
    ax.set_title(f"ρ(x, y=0.5) – t = {times[k]:.3f}")
    return line,

ani1d = FuncAnimation(fig, _update_1d, frames=len(frames_rho),
                      interval=40, blit=True)
vid_1d = f"videos/{prefix}_rho_1d_evol.mp4"
ani1d.save(vid_1d, writer=FFMpegWriter(fps=25, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación 1-D guardada en {vid_1d}")
