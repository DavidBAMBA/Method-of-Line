import numpy as np
import matplotlib.pyplot as plt
import os, glob, re

from config             import NGHOST
from equations          import Euler1D
from initial_conditions import leblanc_shock_tube_1d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
cases = {
    "low": 200,
    "high": 2000
}
xmin, xmax = 0.0, 9.0
tf         = 6.0
cfl        = 0.1
gamma      = 5.0 / 3.0
limiter    = "mp5"
riemann    = "hllc"

# ═════════════════════════════ simulaciones  ══════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

results = {}

for tag, Nx in cases.items():
    print(f"\n[INFO] Ejecutando {tag.upper()} con Nx = {Nx}")
    x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
    init       = leblanc_shock_tube_1d(x_phys, x0=3.0)
    U0, _      = create_U0(nvars=3, shape_phys=(Nx,), initializer=init)
    eq         = Euler1D(gamma=gamma)

    recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
    riem  = lambda UL, UR, eq_, axis=None: solve_riemann(UL, UR, eq_, axis, solver=riemann)

    prefix = f"leblanc_{tag}"
    RK4(dUdt_func=dUdt,
        t0=0.0, U0=U0, tf=tf,
        dx=dx, dy=None,
        equation=eq,
        reconstruct=recon,
        riemann_solver=riem,
        x=x_phys, y=None,
        bc_x=("outflow", "outflow"),
        reflect_idx=[1],
        cfl=cfl,
        save_every=99999,
        filename=prefix,
        reconst=limiter)

    # Leer CSV final
    pattern = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"),
                     key=lambda f: int(re.search(r"_(\d+)\.csv$", f).group(1)))
    dat = np.loadtxt(pattern[-1], delimiter=",", skiprows=1)
    x     = dat[:, 0]
    rho   = dat[:, 1]
    v     = dat[:, 2]
    P     = dat[:, 3]
    e_int = P / ((gamma - 1.0) * rho)
    results[tag] = (x, rho, v, P, e_int)

# ══════════════════════  Figura 2x2 con estilo exacto  ═════════════════════
x_lo, rho_lo, v_lo, P_lo, e_lo = results["low"]
x_hi, rho_hi, v_hi, P_hi, e_hi = results["high"]

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# ρ
axs[0, 0].plot(x_lo, rho_lo, 'o', ms=3, label='N=200')
axs[0, 0].plot(x_hi, rho_hi, '-', lw=2, label='N=2000')
axs[0, 0].set_ylabel(r"$\rho$")
axs[0, 1].set_xlim(0.0, 1.0)
axs[0, 0].legend()

# v
axs[0, 1].plot(x_lo, v_lo, 'o', ms=3)
axs[0, 1].plot(x_hi, v_hi, '-', lw=2)
axs[0, 1].set_xlim(0.0, 1.0)
axs[0, 1].set_ylabel("v")

# P
axs[1, 0].plot(x_lo, P_lo, 'o', ms=3)
axs[1, 0].plot(x_hi, P_hi, '-', lw=2)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("P")

# e_int
axs[1, 1].plot(x_lo, e_lo, 'o', ms=3)
axs[1, 1].plot(x_hi, e_hi, '-', lw=2)
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel(r"$e_{\rm int}$")

fig.suptitle(f"LeBlanc 1-D • {limiter.upper()} • t = {tf}", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_png = f"videos/leblanc_{limiter}.png"
plt.savefig(out_png, dpi=300)
plt.close(fig)

print(f"[INFO] Comparación guardada en {out_png}")
