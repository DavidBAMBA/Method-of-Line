import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# ---- Módulos propios ----
from equations import Burgers2D
from initial_conditions import sinusoidal_burgers_2d
from boundary import periodic
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_2d, create_U0

# ------- Parámetros de la simulación -------
Nx, Ny = 200, 200
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tf     = 0.5
cfl    = 0.1
limiter = "minmod"

# ------- Malla y condición inicial -------
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
U0_array = sinusoidal_burgers_2d(X, Y, kx=2*np.pi, ky=2*np.pi, amp=1.0)
initializer = lambda U: U0_array
U0 = create_U0(nvars=1, shape=(Nx, Ny), initializer=initializer)

equation = Burgers2D()

# Funciones auxiliares
bc_func = periodic
recon   = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

# ------- Evolución -------
sol = RK4(dUdt_func=dUdt,
          boundary_func=bc_func,
          t0=0.0, U0=U0, tf=tf,
          dx=dx, dy=dy,
          equation=equation,
          reconstruct=recon,
          riemann_solver=solve_riemann,
          cfl=cfl)[1]  # ignoramos el tiempo


# === Proyección 2D → 1D por promedio sobre y ===
u0_proj = np.mean(sol[0, 0], axis=1)     # u(x, t=0)
uf_proj = np.mean(sol[-1, 0], axis=1)    # u(x, t=tf)

plt.figure(figsize=(8,4))
plt.plot(x, u0_proj, label="t = 0")
plt.plot(x, uf_proj, label=f"t = {tf}")
plt.xlabel("x")
plt.ylabel(r"$\langle u(x,t) \rangle_y$")
plt.title("Proyección de Burgers 2D sobre x")
plt.legend()
plt.tight_layout()
plt.savefig("videos/burgers_2d_projection_x.png")
plt.show()

# ------- Animación 3D -------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(sol.min() - 0.1, sol.max() + 0.1)
ax.set_title("Burgers 2D – evolución")
surf = [ax.plot_surface(X, Y, sol[0, 0], cmap='viridis')]

def update(frame_idx):
    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, sol[frame_idx, 0], cmap='viridis')
    ax.set_title(f"Frame {frame_idx}")
    return surf

ani = FuncAnimation(fig, update, frames=len(sol), interval=40, blit=False)
video_path = "videos/burgers_2d_3d_evolution.mp4"
writer = FFMpegWriter(fps=100, bitrate=1800)
ani.save(video_path, writer=writer)
plt.close(fig)

print(f"\n Video guardado como: {video_path}")
