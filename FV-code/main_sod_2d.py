import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# ── módulos propios ───────────────────────────────────────────
from equations     import Euler2D
from initial_conditions import sod_shock_tube_1d
from solver        import RK4, dUdt
from reconstruction import reconstruct
from riemann       import solve_riemann
from utils         import create_mesh_2d, create_U0
from boundary      import dirichlet_sod, periodic          # fija extremos x
# ──────────────────────────────────────────────────────────────

# ── parámetros ────────────────────────────────────────────────
Nx, Ny   = 400, 100
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tf, cfl    = 0.2, 0.5
limiter    = "minmod"
gamma      = 1.4
# ──────────────────────────────────────────────────────────────

# ── malla y condición inicial ────────────────────────────────
x, y, dx, dy = create_mesh_2d(xmin, xmax, Nx, ymin, ymax, Ny)
init1d = sod_shock_tube_1d(x, x0=0.5)
U1d    = init1d(np.zeros((3, Nx)))            # (ρ,ρv,E)

U0_4 = np.zeros((4, Nx, Ny))
U0_4[0:2] = np.repeat(U1d[0:2, :, None], Ny, axis=2)  # ρ, ρv_x
U0_4[2]   = 0.0                                       # ρv_y = 0
rho = U0_4[0]
vx  = U0_4[1] / rho
P   = np.where(x[:, None] < 0.5, 1.0, 0.1)
U0_4[3] = P/(gamma-1) + 0.5*rho*vx**2                 # E

U0 = create_U0(nvars=4, shape=(Nx, Ny), initializer=lambda U: U0_4)
equation = Euler2D(gamma)

# ── reconstrucción y BC ──────────────────────────────────────
recon = lambda U, dx, axis: reconstruct(U, dx, limiter=limiter, axis=axis)

def periodic_y(U):
    """periodicidad solo en y (mantiene extremos x)."""
    U[:, :, 0]  = U[:, :, -2]
    U[:, :, -1] = U[:, :,  1]
    return U

def bc(U):
    U = dirichlet_sod(U)   # extremos x fijos
    U = periodic_y(U)      # extremos y periódicos
    return U

# ── integración temporal ─────────────────────────────────────
os.makedirs("videos", exist_ok=True)

times, sol = RK4(dUdt, bc, 0.0, U0, tf,
                 dx, dy, equation, recon, solve_riemann, cfl)

# ── proyección 1-D (promedio en y) y animación ───────────────
rho_series = sol[:, 0].mean(axis=2)       # (nt, Nx)
mom_series = sol[:, 1].mean(axis=2)
v_series   = mom_series / rho_series
E_series   = sol[:, 3].mean(axis=2)
P_series   = (gamma-1)*(E_series - 0.5*rho_series*v_series**2)

# gráfico final
plt.figure(figsize=(10,6))
plt.plot(x, rho_series[-1], label='ρ')
plt.plot(x, v_series[-1],   label='v')
plt.plot(x, P_series[-1],   label='P')
plt.xlabel("x"); plt.ylabel("magnitud")
plt.title(f"Sod 2-D (promedio en y) – t = {tf}")
plt.legend(); plt.tight_layout()
plt.savefig("videos/sod_2d_projection_avg_1d.png")
plt.show()

# animación
fig, ax = plt.subplots(figsize=(10,5))
l_rho, = ax.plot(x, rho_series[0], label='ρ')
l_v,   = ax.plot(x, v_series[0],   label='v')
l_P,   = ax.plot(x, P_series[0],   label='P')
ax.set_xlim(xmin, xmax)
ax.set_ylim(0, max(rho_series.max(), P_series.max())*1.1)
ax.set_xlabel("x"); ax.set_ylabel("magnitud")
ax.set_title("Sod 2-D – evolución (promedio en y)")
ax.legend()

def update(i):
    l_rho.set_ydata(rho_series[i])
    l_v.set_ydata(v_series[i])
    l_P.set_ydata(P_series[i])
    ax.set_title(f"Sod 2-D – t = {times[i]:.3f}")
    return l_rho, l_v, l_P

ani = FuncAnimation(fig, update, frames=len(times), interval=40, blit=True)
writer = FFMpegWriter(fps=25, bitrate=1800)
ani.save("videos/sod_2d_projection_avg_1d.mp4", writer=writer)
plt.close(fig)

print("\n Gráfica y animación guardadas en carpeta 'videos/'")
