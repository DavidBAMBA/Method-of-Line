# main_complex_advection_1d_comparison.py
"""
Comparación de reconstructores para la advección escalar 1D con perfil complejo.
Se evalúa la solución numérica final para varios reconstructores contra la solución exacta.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re

# ── módulos del proyecto ─────────────────────────────────────────────────
from config             import NGHOST
from equations          import Advection1D
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0
from initial_conditions import complex_advection_1d
from write              import setup_data_folder

# ═════════════════════════════════ PARÁMETROS ═════════════════════════════
Nx      = 400
xmin, xmax = -1.0, 1.0
tf      = 500.0
cfl     = 0.1
a       = 1.0
solver  = "exact"
limiters = [ "mp5", "weno5", "wenoz"]
prefix  = "complex_advection_compare"

# ══════════════════════════════ MALLA + INICIAL ═══════════════════════════
x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
L = xmax - xmin

initializer = complex_advection_1d(x_phys)
U0, phys = create_U0(nvars=1, shape_phys=(Nx,), initializer=initializer)
equation = Advection1D(a=a)
U0_ref = U0.copy()

# ═════════════════════════════ SIMULACIONES ══════════════════════════════
os.makedirs("videos", exist_ok=True)
setup_data_folder("data")

sol_final = {}

for lim in limiters:
    recon = lambda U, d, axis=None: reconstruct(U, d, limiter=lim, axis=axis, bc_x="periodic")
    riem  = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=solver)

    RK4(dUdt_func=dUdt,
        t0=0.0, U0=U0_ref.copy(), tf=tf,
        dx=dx, dy=None,
        equation=equation,
        reconstruct=recon,
        riemann_solver=riem,
        x=x_phys, y=None,
        bc_x=("periodic", "periodic"),
        cfl=cfl,
        save_every=10000,   # solo estado final
        filename=f"{prefix}_{lim}",
        reconst=lim)

    # Leer el último CSV generado
    pat = f"data/{prefix}_{lim}_*.csv"
    csv_files = sorted(glob.glob(pat),
                       key=lambda f: int(re.search(r"_(\d{5})\.csv$", f).group(1)))
    
    data = np.loadtxt(csv_files[-1], delimiter=",", skiprows=1)
    u_final = data[:, 1]
    sol_final[lim] = u_final

# ═════════════════════ SOLUCIÓN EXACTA EN t = tf ═════════════════════════
shift_x = ((x_phys - a * tf - xmin) % L) + xmin
u_exact = complex_advection_1d(shift_x)(np.zeros((1, Nx)))[0]

# ══════════════════════ GRÁFICA COMPARATIVA FINAL ════════════════════════
plt.figure(figsize=(10, 5))
for lim, style in zip(limiters, ["--", "-.", "-"]):
    plt.plot(x_phys, sol_final[lim], style, label=f"{lim}")
plt.plot(x_phys, u_exact, "k:", lw=1.5, label="Exacta")

plt.xlabel("x"); plt.ylabel("u(x, t)")
plt.title("Advección 1D compleja – comparación reconstructores vs exacta")
plt.legend(); plt.tight_layout()

cmp_path = f"videos/{prefix}_comparison.png"
plt.savefig(cmp_path, dpi=300)
plt.close()
print(f"[INFO] Gráfica comparativa guardada en {cmp_path}")
