# main_advection_complex.py

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
Nx     = 200
xmin   = -1.0
xmax   = 1.0
tf     = 2.0
cfl    = 0.1

# === Malla y condición inicial ===
x, dx = create_mesh_1d(xmin, xmax, Nx)
initializer = complex_advection_1d(x)
U0 = create_U0(nvars=1, shape=(Nx,), initializer=initializer)

equation = Advection1D(a=1.0)
bc_func = periodic

# --- Guardamos la condición inicial de referencia ---
U0_ref = U0.copy()

# === Simulaciones para diferentes limitadores ===
os.makedirs("videos", exist_ok=True)

limiters = ["minmod", "mc", "mp5"]
sol_final = {}  # Diccionario: limiter → u(x, tf)

for lim in limiters:
    recon = lambda U, dx, axis: reconstruct(U, dx, limiter=lim, axis=axis)

    _, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc_func,
                 t0=0.0, U0=U0_ref.copy(), tf=tf,
                 dx=dx, dy=None,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

    sol_final[lim] = sol[-1, 0]  # Solo guardar el estado final de cada simulación

# === Solución exacta ========================================
# === Solución exacta (desplazamiento a·tf, con a = 1) =========
a = 1.0
shift = a * tf
L = xmax - xmin

# desplazo hacia atrás y aplico periodicidad
x0 = ((x - shift - xmin) % L) + xmin

# construyo el initializer sobre x0 y lo evalúo
init_exact = complex_advection_1d(x0)
U_exact   = init_exact(np.zeros((1, Nx)))   # shape (1,Nx)
u_exact   = U_exact[0]

# === Gráfica comparativa ======================================
plt.figure(figsize=(10, 5))
for lim, style in zip(limiters, ["--", "-.", "-"]):
    plt.plot(x, sol_final[lim], style, label=f"{lim}")
plt.plot(x, u_exact, "k:", lw=1.5, label="exacta t = tf")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.title("Advección compleja – comparación reconstructores vs exacta")
plt.legend(); plt.tight_layout()
plt.savefig("videos/complex_advection_comparison.png")
plt.show()


# === Animación usando MP5 ===
# (opcional: solo para ver cómo evoluciona la mejor reconstrucción)

recon = lambda U, dx, axis: reconstruct(U, dx, limiter="mp5", axis=axis)
_, sol = RK4(dUdt_func=dUdt,
             boundary_func=bc_func,
             t0=0.0, U0=U0_ref.copy(), tf=tf,
             dx=dx, dy=None,
             equation=equation,
             reconstruct=recon,
             riemann_solver=solve_riemann,
             cfl=cfl)

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x, sol[0, 0])
ax.set_ylim(-0.2, 1.2)
ax.set_xlim(xmin, xmax)
ax.set_title("Evolución de advección - MP5")
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")

def update(frame):
    line.set_ydata(frame[0])
    return line,

ani = FuncAnimation(fig, update, frames=sol, interval=40, blit=True)
writer = FFMpegWriter(fps=100, bitrate=1800)
ani.save("videos/complex_advection_evolution.mp4", writer=writer)
plt.close()

print("\n Gráfica comparativa y animación guardadas en 'videos/'")
