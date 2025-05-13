# main_shu_osher_compare.py  ──────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re

from config             import NGHOST
from equations          import Euler1D
from initial_conditions import shu_osher_1d
from solver             import RK4, dUdt
from reconstruction     import reconstruct
from riemann            import solve_riemann
from utils              import create_mesh_1d, create_U0
from write              import setup_data_folder

# ═════════════════════════════════  parámetros  ═══════════════════════════
cases = {
    "MP5_300": 300,
    "MP5_2000": 2000
}
xmin, xmax = -5.0, 5.0
tf         = 1.8
cfl        = 0.1
gamma      = 1.4
limiter    = "mp5"
riemann    = "hllc"

# ═════════════════════════════ simulaciones  ══════════════════════════════
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

results = {}

for tag, Nx in cases.items():
    print(f"\n[INFO] Ejecutando {tag} con Nx = {Nx}")
    x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
    init       = shu_osher_1d(x_phys)
    U0, _      = create_U0(nvars=3, shape_phys=(Nx,), initializer=init)
    eq         = Euler1D(gamma=gamma)

    recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
    riem  = lambda UL, UR, eq_, axis=None: solve_riemann(UL, UR, eq_, axis, solver=riemann)

    prefix = f"shu_{tag}"
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
        save_every=99999,  # solo guardar último paso
        filename=prefix,
        reconst=limiter)

    # Cargar CSV final
    pattern = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"),
                     key=lambda f: int(re.search(r"_(\d+)\.csv$", f).group(1)))
    dat = np.loadtxt(pattern[-1], delimiter=",", skiprows=1)
    rho = dat[:, 1]
    results[tag] = (x_phys, rho)

# ═════════════════════════════  ploteo comparativo  ═══════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot global
for tag, (x, rho) in results.items():
    ax1.plot(x, rho, label=tag.replace("_", " "), lw=1.2)
ax1.set_title("Shu–Osher – perfil global")
ax1.set_xlabel("x"); ax1.set_ylabel("Densidad")
ax1.legend(); 

# Subplot con zoom
for tag, (x, rho) in results.items():
    ax2.plot(x, rho, label=tag.replace("_", " "), lw=1.2)
ax2.set_xlim(0.0, 2.5)
ax2.set_ylim(2.8, 4.75)
ax2.set_title("Shu–Osher – perfil local")
ax2.set_xlabel("x"); ax2.set_ylabel("Densidad")
ax2.legend()

plt.tight_layout()
plt.savefig("videos/shu_osher_compare_rho.png", dpi=300)
plt.show()
print("[INFO] Comparación guardada en videos/shu_osher_compare_rho.png")
