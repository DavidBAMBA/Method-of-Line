import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from mpl_toolkits.mplot3d import Axes3D  # necesario para proyecciones 3D


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
tf = 0.25
cfl = 0.4
limiter = "mc"
gamma = 1.4

# === Malla y condiciones iniciales ===
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

U_init = explosion_problem_2d(X, Y)
initializer = lambda U: U_init
U0 = create_U0(nvars=4, shape=(Nx, Ny), initializer=initializer)

equation = Euler2D(gamma=gamma)
recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)
bc_func = extrapolate  # frontera abierta

# === Simulación ===
os.makedirs("videos", exist_ok=True)

times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=dy,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# === Visualización: densidad final ===
rho = sol[-1, 0]  # densidad
fig, ax = plt.subplots()
im = ax.imshow(rho, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='plasma')
plt.colorbar(im, label='Densidad')
ax.set_title("Explosion 2D – Estado final")
plt.tight_layout()
plt.savefig("videos/explosion_2d_density_final.png")
plt.show()

# === Corte 1D de densidad a lo largo de y = 1.0 ===
j = np.argmin(np.abs(y - 1.0))  # Índice más cercano a y=1
rho_final = sol[-1, 0, :, j]    # Densidad en y=1 al tiempo final

plt.figure(figsize=(5, 4))
plt.plot(x, rho_final, color='green', linewidth=2)
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Corte de densidad en y = 1.0")
plt.xlim(xmin, xmax)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig("videos/explosion_density_slice_y1.png")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Coordenadas de malla
X3D, Y3D = np.meshgrid(x, y, indexing='ij')
surf = [ax.plot_surface(X3D, Y3D, sol[0, 0], cmap='viridis')]

ax.set_zlim(0, sol[:, 0].max() * 1.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("rho")
ax.set_title("Explosión 2D - evolución de la densidad")

def update(frame):
    surf[0].remove()
    surf[0] = ax.plot_surface(X3D, Y3D, sol[frame, 0], cmap='viridis')
    ax.set_title(f"t = {frame * tf / len(sol):.3f}")
    return surf

ani = FuncAnimation(fig, update, frames=len(sol), interval=40)

video_path_3d = "videos/explosion_density_3d.mp4"
writer = FFMpegWriter(fps=25, bitrate=1800)
ani.save(video_path_3d, writer=writer)
plt.close(fig)

print(f"\n Animación 3D guardada como: {video_path_3d}")

""" 
# === Animación ===
fig, ax = plt.subplots()
im = ax.imshow(sol[0, 0], origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='plasma', vmin=rho.min(), vmax=rho.max())
ax.set_title("Explosion 2D – Densidad")
plt.colorbar(im)

def update(frame):
    im.set_data(sol[frame, 0])
    return im,

ani = FuncAnimation(fig, update, frames=len(sol), interval=50, blit=True)
video_path = "videos/explosion_2d_density_evolution.mp4"
writer = FFMpegWriter(fps=25, bitrate=1800)
ani.save(video_path, writer=writer)
plt.close(fig)

print(f"\n Video guardado como: {video_path}") """
