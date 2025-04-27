import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Módulos propios ---
from equations import Euler2D
from initial_conditions import explosion_problem_2d
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_2d, create_U0
from boundary import extrapolate

# === Parámetros ===
Nx, Ny = 400, 400
xmin, xmax = 0.0, 2.0
ymin, ymax = 0.0, 2.0
tf = 0.15
cfl = 0.4
solver = "hll"  # o "hll"
gamma = 1.4
limiters = ["mc", "mp5"]  # vamos a comparar estos dos

# === Malla y condiciones iniciales ===
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Condición inicial
U_init = explosion_problem_2d(X, Y)
initializer = lambda U: U_init

# Estructura para guardar soluciones
rho_finals = {}   # densidad final para cada limiter
rho_slices = {}   # corte 1D en y=1 para cada limiter

os.makedirs("videos", exist_ok=True)

# === Loop sobre los limiters ===
for limiter in limiters:
    print(f"\nSimulando con {limiter.upper()}...")

    U0 = create_U0(nvars=4, shape=(Nx, Ny), initializer=initializer)
    equation = Euler2D(gamma=gamma)

    recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)
    bc_func = extrapolate
    riemann_solver = lambda UL, UR, eq, axis: solve_riemann(UL, UR, eq, axis, solver=solver)

    # --- correr simulación ---
    times, sol = RK4(dUdt_func=dUdt,
                     boundary_func=bc_func,
                     t0=0.0, U0=U0, tf=tf,
                     dx=dx, dy=dy,
                     equation=equation,
                     reconstruct=recon,
                     riemann_solver=solve_riemann,
                     cfl=cfl)

    # --- guardar resultados ---
    rho_final = sol[-1, 0]   # densidad final
    rho_finals[limiter] = rho_final

    # corte 1D en y=1.0
    j = np.argmin(np.abs(y - 1.0))
    rho_slice = rho_final[:, j]
    rho_slices[limiter] = rho_slice

# === Visualización 1: Densidades finales 2D (lado a lado) ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for ax, limiter in zip(axs, limiters):
    im = ax.imshow(rho_finals[limiter], origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap='plasma', vmin=0, vmax=np.max(rho_finals["mc"]))
    ax.set_title(f"Densidad final – {limiter.upper()}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label="Densidad")
plt.tight_layout()
plt.savefig("videos/explosion_comparison_density_2D.png")
plt.show()

# === Visualización 2: Corte 1D comparando MC vs MP5 ===
plt.figure(figsize=(7,5))
plt.plot(x, rho_slices["mc"],  label="MC", linewidth=2)
plt.plot(x, rho_slices["mp5"], label="MP5", linestyle="--", linewidth=2)
plt.xlabel("x")
plt.ylabel("Densidad en y = 1.0")
plt.title("Comparación de corte de densidad (y=1)")
plt.legend()
plt.xlim(xmin, xmax)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig("videos/explosion_comparison_density_slice_y1.png")
plt.show()

print("\n✅ Comparaciones guardadas en carpeta 'videos/'")
