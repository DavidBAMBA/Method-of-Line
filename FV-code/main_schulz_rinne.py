import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

from equations import Euler2D
from initial_conditions import schulz_rinne_2d
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_2d, create_U0
from boundary import extrapolate

# === Parámetros ===
Nx, Ny = 400, 400
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tf   = 0.25
cfl  = 0.3
limiter = "mc"

# === Malla y condición inicial ===
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
U0_array = schulz_rinne_2d(x, y)
initializer = lambda U: U0_array
U0 = create_U0(nvars=4, shape=(Nx, Ny), initializer=initializer)
equation = Euler2D(gamma=1.4)

# === Funciones auxiliares ===
recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)
bc_func = extrapolate

# === Evolución ===
os.makedirs("videos", exist_ok=True)
times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=dy,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# === Contorno de densidad final ===
rho = sol[:, 0]  # shape: (nt, Nx, Ny)

plt.figure(figsize=(6, 5))
plt.contour(rho[-1], levels=30, colors='black', extent=[xmin, xmax, ymin, ymax])
plt.xlabel("x"); plt.ylabel("y")
plt.title("Contornos de densidad – Schulz-Rinne")
plt.tight_layout()
plt.savefig("videos/schulz_rinne_density_contour.png")
plt.close()

# === Imagen de densidad coloreada ===
plt.imshow(rho[-1], origin='lower', extent=[xmin, xmax, ymin, ymax],
           cmap='viridis')
plt.title("Densidad final – Schulz-Rinne")
plt.colorbar(label="rho")
plt.tight_layout()
plt.savefig("videos/schulz_rinne_density_color.png")
plt.close()

""" # === Animación ===
fig, ax = plt.subplots()
im = ax.imshow(rho[0], origin='lower', extent=[xmin, xmax, ymin, ymax],
               cmap='viridis', vmin=rho.min(), vmax=rho.max())
ax.set_title("Evolución de la densidad – Schulz-Rinne")
plt.colorbar(im, ax=ax)

def update(frame):
    im.set_data(rho[frame])
    ax.set_title(f"Densidad – t ≈ {frame * tf / len(rho):.3f}")
    return [im]

ani = FuncAnimation(fig, update, frames=len(rho), interval=50, blit=True)
ani.save("videos/schulz_rinne_evolution.mp4", writer=FFMpegWriter(fps=30, bitrate=1800))
plt.close(fig)

print(" Simulación y visualizaciones guardadas en carpeta 'videos'")
 """

