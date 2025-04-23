import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---- Módulos propios ----
from equations          import Burgers1D
from initial_conditions import gaussian_burgers_1d
from boundary           import periodic
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0

# ------- Parámetros de la simulación -------
Nx      = 400
xmin, xmax = 0.0, 1.0
tf      = 2.5
cfl     = 0.1
limiter = "minmod"   # minmod | mc | superbee

# ------- Malla y condición inicial -------
x, dx          = create_mesh_1d(xmin, xmax, Nx)
initializer    = gaussian_burgers_1d(x, center=0.3, width=0.05, amp=1.0)
U0             = create_U0(nvars=1, shape=(Nx,), initializer=initializer)

equation       = Burgers1D()

# Funciones auxiliares
bc_func = periodic
recon   = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

# ------- Evolución -------
times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=None,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# ------- Gráfica inicial y final -------
plt.figure(figsize=(8,4))
plt.plot(x, sol[0, 0], label="t = 0")
plt.plot(x, sol[-1, 0], label=f"t = {tf}")
plt.title("Burgers 1D – pulso gaussiano")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.legend(); plt.tight_layout()
plt.show()

# ---------- animación ----------
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot(x, sol[0, 0])
ax.set_xlim(xmin, xmax)
ax.set_ylim(sol.min()-0.1, sol.max()+0.1)
ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
ax.set_title("Burgers 1D – evolución")

def update(frame):
    line.set_ydata(frame[0])
    return line,

ani = FuncAnimation(fig, update, frames=sol, interval=40, blit=True)

writer = FFMpegWriter(fps=100, bitrate=1800)
video_name = "videos/burgers_1d_evolution.mp4"
ani.save(video_name, writer=writer)
plt.close(fig)

print(f"\n Video guardado como {video_name}")
