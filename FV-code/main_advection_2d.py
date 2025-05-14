# main_advection_2d.py  ───────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os, glob, re

# --- módulos del proyecto ------------------------------------------------
from config             import NGHOST
from equations          import Advection2D
from initial_conditions import gaussian_advection_2d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_2d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
Nx, Ny        = 200, 200
xmin, xmax    = 0.0, 1.0
ymin, ymax    = 0.0, 1.0
tf            = 1.0
cfl           = 0.3
limiter       = "minmod"
riemann_name  = "exact"
prefix        = "advection2d"

# ═══════════════════════════════ malla + IC ═══════════════════════════════
x_phys, y_phys, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X,    Y                = np.meshgrid(x_phys, y_phys, indexing='ij')

U_phys_init = gaussian_advection_2d(X, Y, center=(0.3, 0.3), width=0.05)  # (1, Nx, Ny)
initializer = lambda U: U.__setitem__(slice(None), U_phys_init)
U0, phys    = create_U0(nvars=1, shape_phys=(Nx, Ny), initializer=initializer)

equation = Advection2D(ax=1.0, ay=1.0)
recon    = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riemann  = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis,
                                                       solver=riemann_name)

# ═════════════════════════════ simulación ════════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

print(f"[INFO] Advección 2D • limiter={limiter.upper()} • solver={riemann_name.upper()} • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=dy,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riemann,
    x=x_phys, y=y_phys,
    bc_x=("periodic", "periodic"),
    bc_y=("periodic", "periodic"),
    cfl=cfl,
    save_every=10,
    filename=prefix,
    reconst=limiter)

# ═══════════════════════════  carga de CSVs  ══════════════════════════════
def _step(fname):
    return int(re.search(r"_(\d{5})\.csv$", fname).group(1))

csv_files = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"), key=_step)

frames_u, times = [], []
for f in csv_files:
    data   = np.loadtxt(f, delimiter=",", skiprows=1)
    t_real = data[0, -1]
    u      = data[:, 2].reshape((Nx, Ny))  # columnas: x, y, u, t
    frames_u.append(u)
    times.append(t_real)

frames_u = np.asarray(frames_u)
times    = np.array(times)

# ═════════════════════════════ Animación 3D ═══════════════════════════════
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
X3D, Y3D = np.meshgrid(x_phys, y_phys, indexing='ij')

surf = [ax.plot_surface(X3D, Y3D, frames_u[0], cmap='viridis')]
ax.set_zlim(frames_u.min() - 0.1, frames_u.max() + 0.1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u(x,y)")
ax.set_title("Advección 2D – evolución")

def _update(k):
    surf[0].remove()
    surf[0] = ax.plot_surface(X3D, Y3D, frames_u[k], cmap='viridis')
    ax.set_title(f"t = {times[k]:.3f}")
    return surf

ani = FuncAnimation(fig, _update, frames=len(frames_u), interval=40, blit=False)
vid_3d = f"videos/{prefix}_3d_evolution.mp4"
ani.save(vid_3d, writer=FFMpegWriter(fps=25, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación 3D guardada en {vid_3d}")
