import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from mpl_toolkits.mplot3d import Axes3D


# === Módulos del código ===
from equations import Advection2D
from initial_conditions import gaussian_advection_2d
from boundary import periodic
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_2d, create_U0

# === Parámetros ===
Nx, Ny = 200, 200
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tf   = 1.0
cfl  = 0.3
limiter = "minmod"

# === Malla y condiciones iniciales ===
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)

X, Y = np.meshgrid(x, y, indexing='ij')

initializer_array = gaussian_advection_2d(X, Y, center=(0.3, 0.3), width=0.05)
initializer = lambda U: initializer_array  # convierte en función compatible
U0 = create_U0(nvars=1, shape=(Nx, Ny), initializer=initializer)


equation = Advection2D(ax=1.0, ay=1.0)

# === Condición de frontera y reconstrucción ===
bc_func = periodic
recon   = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

# === Evolución temporal ===
times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=dy,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# Crear figura 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(sol.min() - 0.1, sol.max() + 0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y, t)')
ax.set_title('Advección escalar 2D')

# Superficie inicial como lista mutable
surf = [ax.plot_surface(X, Y, sol[0, 0], cmap='viridis')]

def update(frame_idx):
    surf[0].remove()  # elimina la superficie anterior
    surf[0] = ax.plot_surface(X, Y, sol[frame_idx, 0], cmap='viridis')
    ax.set_title(f"t = {times[frame_idx]:.2f}")
    return surf

ani = FuncAnimation(fig, update, frames=len(times), interval=40)

# Guardar animación
os.makedirs("videos", exist_ok=True)
video_path = "videos/advection_2d_3d_evolution.mp4"
writer = FFMpegWriter(fps=100, bitrate=1800)
ani.save(video_path, writer=writer)
plt.close(fig)

print(f"\n Animación 3D guardada como: {video_path}")
