import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# === Módulos del código ===
from equations import Advection1D
from initial_conditions import gaussian_advection_1d
from boundary import periodic
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_1d, create_U0

# === Parámetros ===
Nx     = 400
xmin   = 0.0
xmax   = 1.0
tf     = 1.0
cfl    = 0.3
limiter = "minmod"      # minmod | mc | superbee

# === Crear malla y condiciones iniciales ===
x, dx          = create_mesh_1d(xmin, xmax, Nx)
initializer    = gaussian_advection_1d(x)
U0             = create_U0(nvars=1, shape=(Nx,), initializer=initializer)
equation       = Advection1D(a=1.0)

# === Función de frontera y reconstrucción ===
bc_func = periodic
recon   = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

# === Correr simulación ===
times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=None,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# === Visualización ===
plt.figure(figsize=(8,4))
plt.plot(x, sol[0, 0], label="t=0")
plt.plot(x, sol[-1, 0], label=f"t={tf}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Advección escalar 1D")
plt.legend()
plt.tight_layout()
plt.savefig("videos/advection_1d.png")
plt.show()

# ---------- animación ----------
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot(x, sol[0, 0])
ax.set_xlim(xmin, xmax)
ax.set_ylim(sol.min()-0.1, sol.max()+0.1)
ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
ax.set_title("Burgers 1D - evolución")

def update(frame):
    line.set_ydata(frame[0])
    return line,

ani = FuncAnimation(fig, update, frames=sol, interval=40, blit=True)

writer = FFMpegWriter(fps=100, bitrate=1800)
video_path = "videos/advection_1d_evolution.mp4"
ani.save(video_path, writer=writer)
plt.close(fig)

print(f"\n Video guardado como {video_path}")
