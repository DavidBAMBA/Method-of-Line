# main_schulz_rinne_2d.py  ─────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re

# --- módulos del proyecto ------------------------------------------------
from config             import NGHOST
from equations          import Euler2D
from initial_conditions import schulz_rinne_2d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_2d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
Nx, Ny        = 400, 400
xmin, xmax    = 0.0, 1.0
ymin, ymax    = 0.0, 1.0
tf            = 0.25
cfl           = 0.1
limiter       = "mc"
riemann_name  = "hllc"
gamma         = 1.4
prefix        = "schulz_rinne"

# ═══════════════════════════════ malla + IC ═══════════════════════════════
x_phys, y_phys, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X,    Y                = np.meshgrid(x_phys, y_phys, indexing='ij')

U_phys_init = schulz_rinne_2d(X, Y)
initializer = lambda U: U.__setitem__(slice(None), U_phys_init)
U0, phys    = create_U0(nvars=4, shape_phys=(Nx, Ny), initializer=initializer)

equation = Euler2D(gamma=gamma)

recon   = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riemann = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis,
                                                      solver=riemann_name)

# ═════════════════════════════ simulación ════════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

print(f"[INFO] Schulz-Rinne 2D • limiter={limiter.upper()} • solver={riemann_name.upper()} • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=dy,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riemann,
    x=x_phys, y=y_phys,
    bc_x=("outflow", "outflow"),
    bc_y=("outflow", "outflow"),
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
    rho    = data[:, 2]
    rho    = rho.reshape((Nx, Ny))
    frames_rho.append(rho)
    times.append(t_real)

frames_rho = np.asarray(frames_rho)
times      = np.array(times)

# ═══════════════════════════  estado final 2-D  ═══════════════════════════
plt.figure(figsize=(6, 5))
plt.imshow(frames_rho[-1], origin='lower',
           extent=[xmin, xmax, ymin, ymax], cmap='viridis')
plt.colorbar(label='ρ')
plt.title("Schulz-Rinne 2-D – Densidad final")
plt.tight_layout()
png_final = f"videos/{prefix}_density_final.png"
plt.savefig(png_final, dpi=300); plt.close()
print(f"[INFO] Imagen final guardada en {png_final}")

# ══════════════════════  animación 2-D  ═══════════════════════════════════
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(frames_rho[0], origin='lower',
               extent=[xmin, xmax, ymin, ymax],
               cmap='viridis', vmin=frames_rho.min(), vmax=frames_rho.max())
cb = plt.colorbar(im, ax=ax, label="ρ")
ax.set_title("Densidad – Schulz-Rinne")

def _update(frame):
    im.set_data(frames_rho[frame])
    ax.set_title(f"t = {times[frame]:.3f}")
    return [im]

ani = FuncAnimation(fig, _update, frames=len(frames_rho), interval=40, blit=True)
vid_2d = f"videos/{prefix}_density_2d.mp4"
ani.save(vid_2d, writer=FFMpegWriter(fps=25, bitrate=1800))
plt.close(fig)
print(f"[INFO] Animación 2-D guardada en {vid_2d}")
