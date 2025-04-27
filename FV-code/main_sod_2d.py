import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# ── módulos propios ───────────────────────────────────────────
from equations import Euler2D
from initial_conditions import sod_shock_tube_2d
from solver import RK4, dUdt
from reconstruction import reconstruct
from riemann import solve_riemann
from utils import create_mesh_2d, create_U0
from boundary import dirichlet_sod, periodic
# ──────────────────────────────────────────────────────────────

# ── parámetros ────────────────────────────────────────────────
Nx, Ny     = 200, 100
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tf, cfl    = 0.2, 0.5
limiter    = "mp5"      # "mc" o "superbee"
solver     = "hll"     # "hllc" o "hll"
gamma      = 1.4
# ──────────────────────────────────────────────────────────────

# ── malla y condición inicial ────────────────────────────────
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
init2d = sod_shock_tube_2d(x, y, x0=0.5)
U0 = create_U0(nvars=4, shape=(Nx, Ny), initializer=init2d)
equation = Euler2D(gamma)

# ── reconstrucción y condiciones de frontera ─────────────────
recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

def periodic_y(U):
    """Periodicidad solo en y (extremos en x fijos)."""
    U[:, :, 0]  = U[:, :, -2]
    U[:, :, -1] = U[:, :,  1]
    return U

def bc(U):
    U = dirichlet_sod(U)  # extremos x fijos
    U = periodic_y(U)     # extremos y periódicos
    return U

riemann_solver = lambda UL, UR, eq, axis: solve_riemann(UL, UR, eq, axis, solver=solver)

# ── integración temporal ─────────────────────────────────────
os.makedirs("videos", exist_ok=True)

times, sol = RK4(dUdt_func=dUdt,
                 boundary_func=bc,
                 t0=0.0, U0=U0, tf=tf,
                 dx=dx, dy=dy,
                 equation=equation,
                 reconstruct=recon,
                 riemann_solver=solve_riemann,
                 cfl=cfl)

# ── posprocesado: promedio en y ──────────────────────────────
rho_series = sol[:, 0].mean(axis=2)  # (nt, Nx)
momx_series = sol[:, 1].mean(axis=2)
vx_series   = momx_series / rho_series
E_series    = sol[:, 3].mean(axis=2)
P_series    = (gamma-1)*(E_series - 0.5*rho_series*vx_series**2)

# ── gráfico final ────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.plot(x, rho_series[-1], label='ρ (densidad)')
plt.plot(x, vx_series[-1],  label='v (velocidad)')
plt.plot(x, P_series[-1],   label='P (presión)')
plt.xlabel("$x$"); plt.ylabel("Magnitud")
plt.title(f"Sod 2-D (promedio en y) – $t={tf}$")
plt.legend(); plt.tight_layout()
plt.savefig("videos/sod_2d_projection_avg_1d.png")
plt.show()

# ── animación ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
l_rho, = ax.plot(x, rho_series[0], label='ρ')
l_v,   = ax.plot(x, vx_series[0],  label='v')
l_P,   = ax.plot(x, P_series[0],   label='P')

ax.set_xlim(xmin, xmax)
ax.set_ylim(0, max(rho_series.max(), P_series.max()) * 1.1)
ax.set_xlabel("$x$")
ax.set_ylabel("Magnitud")
ax.set_title("Sod 2-D - evolución (promedio en y)")
ax.legend()

def update(frame):
    l_rho.set_ydata(rho_series[frame])
    l_v.set_ydata(vx_series[frame])
    l_P.set_ydata(P_series[frame])
    ax.set_title(f"Sod 2-D - t = {times[frame]:.3f}")
    return l_rho, l_v, l_P

ani = FuncAnimation(fig, update, frames=len(times), interval=40, blit=True)
writer = FFMpegWriter(fps=25, bitrate=1800)
ani.save("videos/sod_2d_projection_avg_1d.mp4", writer=writer)
plt.close(fig)

print("\nGráfica y animación guardadas en carpeta 'videos/' ✅")
