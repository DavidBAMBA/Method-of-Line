import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# --- Módulos propios ---
from equations import Euler1D
from initial_conditions import sod_shock_tube_1d
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_1d, create_U0
from boundary import dirichlet_sod

# === Parámetros ===
Nx     = 200
xmin   = 0.0
xmax   = 1.0
tf     = 0.2
cfl    = 0.3
solver     = "hll"  # "hllc" o "hll"

limiter = "mc"  # minmod | mc | superbee

# === Malla y condición inicial ===
x, dx = create_mesh_1d(xmin, xmax, Nx)
initializer = sod_shock_tube_1d(x, x0=0.5)
U0 = create_U0(nvars=3, shape=(Nx,), initializer=initializer)

equation = Euler1D(gamma=1.4)

# === Reconstrucción y condiciones de frontera ===
bc_func = dirichlet_sod

recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)
riemann_solver = lambda UL, UR, eq, axis: solve_riemann(UL, UR, eq, axis, solver=solver)

# === Evolución temporal ===
times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=None,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# === Extraer densidad, velocidad y presión ===
rho = sol[:, 0]
mom = sol[:, 1]
E   = sol[:, 2]
v   = mom / rho
P   = (equation.gamma - 1.0) * (E - 0.5 * rho * v**2)

# === Plot final ===
os.makedirs("videos", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(x, rho[-1], label='Densidad')
plt.plot(x, v[-1], label='Velocidad')
plt.plot(x, P[-1], label='Presión')
plt.xlabel("x")
plt.ylabel("Variables conservadas")
plt.title("Sod shock tube – estado final")
plt.legend()
plt.tight_layout()
plt.xlim(0,1)
plt.savefig("videos/sod_final.png")
plt.show()

# === Animación ===
fig, ax = plt.subplots(figsize=(10, 5))
line_rho, = ax.plot(x, rho[0], label='ρ')
line_v,   = ax.plot(x, v[0], label='v')
line_P,   = ax.plot(x, P[0], label='P')
ax.set_ylim(0, max(rho.max(), P.max()) * 1.1)
ax.set_xlim(xmin, xmax)
ax.set_title("Sod shock tube – evolución")
ax.set_xlabel("x")
ax.set_ylabel("Valor")
ax.legend()

def update(frame):
    line_rho.set_ydata(rho[frame])
    line_v.set_ydata(v[frame])
    line_P.set_ydata(P[frame])
    return line_rho, line_v, line_P

ani = FuncAnimation(fig, update, frames=len(sol), interval=40, blit=True)
video_path = "videos/sod_1d_evolution.mp4"
writer = FFMpegWriter(fps=100, bitrate=1800)
ani.save(video_path, writer=writer)
plt.close(fig)

print(f"\n Animación y gráfico guardados en: {video_path}")
