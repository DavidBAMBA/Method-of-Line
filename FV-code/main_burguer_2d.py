# main_burgers_2d.py  ─────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os, glob, re

# --- módulos del proyecto ------------------------------------------------
from config             import NGHOST
from equations          import Burgers2D
from initial_conditions import sinusoidal_burgers_2d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_2d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
Nx, Ny        = 200, 200
xmin, xmax    = 0.0, 1.0
ymin, ymax    = 0.0, 1.0
tf            = 0.5
cfl           = 0.1
limiter       = "minmod"
riemann_name  = "exact"      # Burgers tiene solución exacta Riemann
prefix        = "burgers2d"

# ═══════════════════════════════ malla + IC ═══════════════════════════════
x_phys, y_phys, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X,    Y                = np.meshgrid(x_phys, y_phys, indexing='ij')

U_phys_init = sinusoidal_burgers_2d(X, Y, kx=2*np.pi, ky=2*np.pi, amp=1.0)  # (1, Nx, Ny)
initializer = lambda U: U.__setitem__(slice(None), U_phys_init)
U0, phys    = create_U0(nvars=1, shape_phys=(Nx, Ny), initializer=initializer)

equation = Burgers2D()
recon   = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riemann = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis,
                                                      solver=riemann_name)

# ═════════════════════════════ simulación ════════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

print(f"[INFO] Burgers 2-D • limiter={limiter.upper()} • solver={riemann_name.upper()} • NGHOST={NGHOST}")

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
    u      = data[:, 2].reshape((Nx, Ny))  # columna
