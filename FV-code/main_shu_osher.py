import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# --- Módulos propios ---
from equations import Euler1D
from initial_conditions import shu_osher_1d
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_1d, create_U0
from boundary import extrapolate

# === Parámetros ===
Nx     = 3000
xmin   = -5.0
xmax   = 5.0
tf     = 1.8
cfl    = 0.3
limiter = "mc"  # minmod | mc | superbee

# === Malla y condición inicial ===
x, dx = create_mesh_1d(xmin, xmax, Nx)
initializer = shu_osher_1d(x)
U0 = create_U0(nvars=3, shape=(Nx,), initializer=initializer)

equation = Euler1D(gamma=1.4)
bc_func = extrapolate
recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

# === Evolución temporal ===
times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=None,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# === Extraer variables ===
rho = sol[:, 0]
mom = sol[:, 1]
E   = sol[:, 2]
v   = mom / rho
P   = (equation.gamma - 1.0) * (E - 0.5 * rho * v**2)

# === Guardar visualización final ===
os.makedirs("videos", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(x, rho[-1], label='ρ')
plt.plot(x, v[-1], label='v')
plt.plot(x, P[-1], label='P')
plt.xlabel("x")
plt.ylabel("Variables")
plt.title("Shu-Osher - Estado final")
plt.legend()
plt.tight_layout()
plt.xlim(xmin, xmax)
plt.savefig("videos/shu_osher_final.png")
plt.show()

# === Animación ===
fig, ax = plt.subplots(figsize=(10, 5))
line_rho, = ax.plot(x, rho[0], label='rho')
line_v,   = ax.plot(x, v[0], label='v')
line_P,   = ax.plot(x, P[0], label='P')
ax.set_ylim(0, max(rho.max(), P.max()) * 1.2)
ax.set_xlim(xmin, xmax)
ax.set_title("Shu-Osher - evolución")
ax.set_xlabel("x")
ax.set_ylabel("Valor")
ax.legend()

def update(frame):
    line_rho.set_ydata(rho[frame])
    line_v.set_ydata(v[frame])
    line_P.set_ydata(P[frame])
    return line_rho, line_v, line_P

ani = FuncAnimation(fig, update, frames=len(sol), interval=40, blit=True)
video_path = "videos/shu_osher_1d_evolution.mp4"
writer = FFMpegWriter(fps=60, bitrate=1800)
ani.save(video_path, writer=writer)
plt.close(fig)

print(f"\n Video guardado como: {video_path}")
