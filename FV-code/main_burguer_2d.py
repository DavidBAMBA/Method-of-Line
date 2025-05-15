# main_burgers_2d.py
"""
Simulación de Burgers 2D con condiciones periódicas.
Perfil inicial sinusoidal. Se visualiza u(x,y,t).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
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

# ═════════════════════════════════ PARÁMETROS ═════════════════════════════
Nx, Ny        = 100, 100
xmin, xmax    = 0.0, 1.0
ymin, ymax    = 0.0, 1.0
tf            = 0.5
cfl           = 0.1
limiter       = "mp5"
solver        = "exact"
prefix        = "burgers2d"

# ═══════════════════════════════ MALLA + IC ═══════════════════════════════
x_phys, y_phys, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X, Y = np.meshgrid(x_phys, y_phys, indexing='ij')

U_phys_init = sinusoidal_burgers_2d(X, Y, kx=2*np.pi, ky=2*np.pi, amp=1.0)
initializer = sinusoidal_burgers_2d(X,Y) #lambda U: U.__setitem__(slice(None), U_phys_init)
U0, phys = create_U0(nvars=1, shape_phys=(Nx, Ny), initializer=initializer)

equation = Burgers2D()
recon    = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
riemann  = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=solver)

# ═════════════════════════════ SIMULACIÓN ════════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

print(f"[INFO] Burgers 2-D • limiter={limiter.upper()} • solver={solver.upper()} • NGHOST={NGHOST}")

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

# ═════════════════════════════ CARGA DE DATOS ═════════════════════════════
def _step(fname): return int(re.search(r"_(\d{5})\.csv$", fname).group(1))
csv_files = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"), key=_step)

frames_u, times = [], []
for f in csv_files:
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    u = data[:, 2].reshape((Nx, Ny))
    t_real = data[0, -1]
    frames_u.append(u)
    times.append(t_real)

frames_u = np.asarray(frames_u)
times = np.asarray(times)

# ═════════════════════════════ FIGURA FINAL 2D ════════════════════════════
plt.figure(figsize=(6,5))
plt.imshow(frames_u[-1], origin='lower',
           extent=[xmin, xmax, ymin, ymax], cmap='plasma')
plt.colorbar(label='u')
plt.title("Burgers 2-D – Estado final")
plt.tight_layout()
png_final = f"videos/{prefix}_final.png"
plt.savefig(png_final, dpi=300); plt.close()
print(f"[INFO] Imagen final guardada en {png_final}")

# ═════════════════════════════ ANIMACIÓN 3D ═══════════════════════════════
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
X3D, Y3D = np.meshgrid(x_phys, y_phys, indexing='ij')

surf = [ax.plot_surface(X3D, Y3D, frames_u[0], cmap='viridis')]
ax.set_zlim(frames_u.min(), frames_u.max()*1.1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")

def _update(k):
    surf[0].remove()
    surf[0] = ax.plot_surface(X3D, Y3D, frames_u[k], cmap='viridis')
    ax.set_title(f"t = {times[k]:.3f}")
    return surf

ani = FuncAnimation(fig, _update, frames=len(frames_u),
                    interval=40, blit=False)
vid_3d = f"videos/{prefix}_3d.mp4"
ani.save(vid_3d, writer=FFMpegWriter(fps=25, bitrate=1800))
plt.close(fig)

print(f"[INFO] Animación 3-D guardada en {vid_3d}")
