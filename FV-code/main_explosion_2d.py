# main_explosion_2d.py  ────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401 (implic. usado)
import os, glob, re

# --- módulos del proyecto -------------------------------------------------
from config             import NGHOST
from equations          import Euler2D
from initial_conditions import explosion_problem_2d
from solver             import RK4, dUdt                 # ← puedes pasar dUdt_high_order si quieres
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_2d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
Nx, Ny        = 400, 400          # celdas físicas
xmin, xmax    = 0.0, 2.0
ymin, ymax    = 0.0, 2.0
tf            = 0.15
cfl           = 0.4
limiter       = "mp5"
solver        = "hllc"            # "hll" o "hllc"
gamma         = 1.4
prefix        = "explosion2d"

# ═══════════════════════════════ malla + IC ═══════════════════════════════
x_phys, y_phys, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
init = explosion_problem_2d(x_phys, y_phys)              # (4, Nx, Ny)
U0, phys = create_U0(nvars=4, shape_phys=(Nx, Ny), initializer=init)

equation = Euler2D(gamma=gamma)

recon   = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis, bc_x="outflow", bc_y="outflow")
riemann = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis,
                                                      solver=solver)

# ═════════════════════════════ simulación ════════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

print(f"[INFO] Explosión 2-D • limiter={limiter.upper()} • solver={solver.upper()} • NGHOST={NGHOST}")

RK4(dUdt_func=dUdt,                 # usa dUdt_high_order si lo prefieres
    t0=0.0, U0=U0, tf=tf,
    dx=dx, dy=dy,
    equation=equation,
    reconstruct=recon,
    riemann_solver=riemann,
    x=x_phys, y=y_phys,
    bc_x=("outflow", "outflow"),    # fronteras abiertas
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
    rho    = data[:, 2]                   # columnas: x, y, rho, v, P, time
    rho    = rho.reshape((Nx, Ny))
    frames_rho.append(rho)
    times.append(t_real)

frames_rho = np.asarray(frames_rho)
times      = np.array(times)

# ═══════════════════════════  estado final 2-D  ═══════════════════════════
plt.figure(figsize=(6,5))
plt.imshow(frames_rho[-1], origin='lower',
           extent=[xmin, xmax, ymin, ymax], cmap='plasma')
plt.colorbar(label='ρ')
plt.title("Explosión 2-D – Densidad (estado final)")
plt.tight_layout()
png_final = f"videos/{prefix}_density_final.png"
plt.savefig(png_final, dpi=300); plt.close()
print(f"[INFO] Imagen final guardada en {png_final}")

# ══════════════════════  corte 1-D (y = 1.0)  ═════════════════════════════
j = np.argmin(np.abs(y_phys - 1.0))
rho_slice = frames_rho[-1][:, j]

plt.figure(figsize=(6,4))
plt.plot(x_phys, rho_slice, color='tab:green')
plt.xlabel("x"); plt.ylabel("ρ(x, y=1)")
plt.title("Corte de densidad en y = 1.0")
plt.xlim(xmin, xmax); plt.ylim(0, rho_slice.max()*1.1)
plt.tight_layout()
png_slice = f"videos/{prefix}_slice_y1.png"
plt.savefig(png_slice, dpi=300); plt.close()
print(f"[INFO] Corte 1-D guardado en {png_slice}")

# ══════════════════════  animación 3-D  ═══════════════════════════════════
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
X3D, Y3D = np.meshgrid(x_phys, y_phys, indexing='ij')

surf = [ax.plot_surface(X3D, Y3D, frames_rho[0], cmap='viridis')]
ax.set_zlim(0, frames_rho.max()*1.05)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("ρ")

def _update(k):
    surf[0].remove()
    surf[0] = ax.plot_surface(X3D, Y3D, frames_rho[k], cmap='viridis')
    ax.set_title(f"t = {times[k]:.3f}")
    return surf

ani = FuncAnimation(fig, _update, frames=len(frames_rho),
                    interval=40, blit=False)
vid_3d = f"videos/{prefix}_density_3d.mp4"
ani.save(vid_3d, writer=FFMpegWriter(fps=25, bitrate=1800))
plt.close(fig)

print(f"[INFO] Animación 3-D guardada en {vid_3d}")