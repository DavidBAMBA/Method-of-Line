# main_explosion_comparison_2d.py ─────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401 (implícito usado)
import os, glob, re

# --- módulos del proyecto ------------------------------------------------
from config             import NGHOST
from equations          import Euler2D
from initial_conditions import explosion_problem_2d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_2d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
Nx, Ny        = 400, 400
xmin, xmax    = 0.0, 2.0
ymin, ymax    = 0.0, 2.0
tf            = 0.15
cfl           = 0.4
gamma         = 1.4
solver        = "hll"
limiters      = ["mc", "mp5", "weno3", "wenoz"]

setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

# ════════════════════════ malla y condiciones iniciales ══════════════════
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
init         = explosion_problem_2d(x, y)
j_center     = np.argmin(np.abs(y - 1.0))  # para el corte en y = 1

# ══════════════════════════ estructuras para resultados ══════════════════
rho_finals = {}  # densidades finales 2D
rho_slices = {}  # cortes 1D en y = 1.0

# ═════════════════════════════ loop de simulaciones ══════════════════════
for limiter in limiters:
    print(f"[INFO] Simulando con {limiter.upper()}...")

    U0, _ = create_U0(nvars=4, shape_phys=(Nx, Ny), initializer=init)
    equation = Euler2D(gamma=gamma)

    recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis,
                                                bc_x="outflow", bc_y="outflow")
    riemann = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=solver)

    RK4(dUdt_func=dUdt,
        t0=0.0, U0=U0, tf=tf,
        dx=dx, dy=dy,
        equation=equation,
        reconstruct=recon,
        riemann_solver=riemann,
        x=x, y=y,
        bc_x=("outflow", "outflow"),
        bc_y=("outflow", "outflow"),
        cfl=cfl,
        save_every=100,
        filename=f"expl_{limiter}",
        reconst=limiter)

    rho = U0[0]  # estado final actualizado en U0
    rho_finals[limiter] = rho
    rho_slices[limiter] = rho[:, j_center]  # corte en y=1

# ═══════════════════════ visualización 1: mapas de densidad 2D ═══════════
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

vmax_ref = np.max(rho_finals["mc"])  # para normalizar el colormap

for ax, lim in zip(axs, limiters):
    im = ax.imshow(rho_finals[lim], origin="lower",
                   extent=[xmin, xmax, ymin, ymax],
                   cmap="plasma", vmin=0.0, vmax=vmax_ref)
    ax.set_title(f"{lim.upper()}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle("Densidad final – Comparación entre reconstructores", fontsize=14)
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85, label="Densidad")
plt.tight_layout()
plt.savefig("videos/explosion_comparison_density_2D.png", dpi=300)
plt.close()

# ═══════════════════════ visualización 2: corte 1D en y = 1.0 ════════════
plt.figure(figsize=(10, 5))
for lim in limiters:
    plt.plot(x, rho_slices[lim], label=lim.upper(), linewidth=2)

plt.title("Corte de densidad en y = 1.0")
plt.xlabel("x")
plt.ylabel("Densidad")
plt.xlim(xmin, xmax)
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("videos/explosion_comparison_density_slice_y1.png", dpi=300)
plt.close()

print("\n Comparaciones guardadas en 'videos/'")
