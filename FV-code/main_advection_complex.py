import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# --- Módulos propios ---
from equations import Advection1D
from initial_conditions import complex_advection_1d
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_1d, create_U0
from boundary import periodic

# === Parámetros ===
Nx     = 400
xmin   = -1.0
xmax   = 1.0
tf     = 2.0
cfl    = 0.1
limiter = "mp5"  # minmod | mc | superbee | mp5

# === Malla y condición inicial ===
x, dx = create_mesh_1d(xmin, xmax, Nx)
initializer = complex_advection_1d(x)
U0 = create_U0(nvars=1, shape=(Nx,), initializer=initializer)

equation = Advection1D(a=1.0)
bc_func = periodic
recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

# === Evolución temporal ===
os.makedirs("videos", exist_ok=True)

times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=None,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# === Estado inicial y final ===
plt.figure(figsize=(10, 5))
plt.plot(x, sol[0, 0], label='t=0')
plt.plot(x, sol[-1, 0], label=f't={tf}')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Advección de ondas complejas")
plt.legend()
plt.tight_layout()
plt.savefig("videos/complex_advection_final.png")
plt.show()

# === Animación ===
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x, sol[0, 0])
ax.set_ylim(-0.2, 1.2)
ax.set_xlim(xmin, xmax)
ax.set_title("Advección de ondas complejas")
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")

def update(frame):
    line.set_ydata(frame[0])
    return line,

ani = FuncAnimation(fig, update, frames=sol, interval=40, blit=True)
writer = FFMpegWriter(fps=100, bitrate=1800)
ani.save("videos/complex_advection_evolution.mp4", writer=writer)
plt.close(fig)

print("\n Animación y gráfico guardados en 'videos/'")
