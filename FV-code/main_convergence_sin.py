# main_convergence_sin.py
"""
Test de “critical points” (Henrick et al., 2005) con BC periódicas.
u0(x) = sin(π x − sin(π x)/π)  en  x ∈ [−1,1],  velocidad a = 1.
Se evalúa la convergencia para reconstructores MC (2º) y MP5 (5º).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os, glob, re, itertools

# ─── módulos del proyecto ─────────────────────────────────────────────
from config        import NGHOST
from equations     import Advection1D
from initial_conditions import critical_sine_1d   # reutilizamos plantilla
from solver        import RK4F, dUdt
from reconstruction import reconstruct
from riemann        import solve_riemann
from utils          import create_mesh_1d, create_U0
from write          import setup_data_folder



# ─── parámetros globales ──────────────────────────────────────────────
xmin, xmax = -1.0, 1.0
a          = 1.0
tf         = 8.0
cfl        = 0.1                     # sobrado para advección 1-D
solver     = "hllc"                 # flujo exacto = upwind
limiters   = ["mc", "mp5"]
N_list     = [20, 40, 80, 160]

# carpetas
setup_data_folder("data")
os.makedirs("videos", exist_ok=True)

# utilidades ----------------------------------------------------------------
def extract_step(fname):
    m = re.search(r"_(\d{5})\.csv$", fname)
    return int(m.group(1)) if m else -1

def exact_solution(x, t):
    """u(x,t) exacto desplazando la IC con velocidad +1."""
    L = xmax - xmin
    x_shift = ((x - a*t - xmin) % L) + xmin     # envoltura periódica
    return np.sin(np.pi*x_shift - np.sin(np.pi*x_shift)/np.pi)

# ─── bucle sobre (Nx, limiter) ─────────────────────────────────────────
results_L2, results_Linf = {l:[] for l in limiters}, {l:[] for l in limiters}

for limiter, Nx in itertools.product(limiters, N_list):

    # ── malla e IC ───────────────────────────────────────────
    x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
    init       = critical_sine_1d(x_phys)
    U0, phys   = create_U0(nvars=1, shape_phys=(Nx,), initializer=init)

    # ecuación + callbacks
    equation = Advection1D(a=a)
    recon    = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
    riem     = lambda UL, UR, eq, axis=None: solve_riemann(UL, UR, eq, axis, solver=solver)

    prefix   = f"crit_{limiter}_{Nx}"

    print(f"[RUN] {limiter.upper():5s}  Nx={Nx:4d}")
    fixed_step = 0.1*dx*dx
    RK4F(dUdt_func=dUdt,
        t0=0.0, U0=U0, tf=tf,
        dx=dx, dy=None,
        equation=equation,
        reconstruct=recon,
        riemann_solver=riem,
        x=x_phys, y=None,
        bc_x=("periodic","periodic"),
        fixed_step=fixed_step,
        save_every=10,          # solo guarda paso final
        filename=prefix,
        reconst=limiter)

    # ── leer el CSV final ───────────────────────────────────
    csv_file = sorted(glob.glob(f"data/{prefix}_{limiter}_*.csv"),
                      key=extract_step)[-1]
    data   = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    u_num  = data[:,1]
    t_real = data[0,-1]
    u_exact = exact_solution(x_phys, t_real)

    err = u_num - u_exact
    err_L2   = np.sqrt(np.mean(err**2))
    err_Linf = np.max(np.abs(err))
    results_L2[limiter].append((Nx, err_L2))
    results_Linf[limiter].append((Nx, err_Linf))

    # ── animación (num vs exacto) ───────────────────────────
    fig, ax = plt.subplots(figsize=(8,3))
    ax.set(xlim=(xmin,xmax), ylim=(-1.2,1.2),
           xlabel="x", ylabel="u")
    ln_num,   = ax.plot([], [], lw=2, label="Numérico")
    ln_exact, = ax.plot([], [], "k--", label="Exacto")
    ax.legend()

    frames = 200
    times  = np.linspace(0, tf, frames)

    def update(i):
        ti = times[i]
        # solución numérica desplazando array inicial porque advección es lineal
        shift = int(round(ti/dx)) % Nx
        u_num_t = np.roll(U0[0, NGHOST:-NGHOST], shift)
        ln_num.set_data(x_phys, u_num_t)
        ln_exact.set_data(x_phys, exact_solution(x_phys, ti))
        ax.set_title(f"{limiter.upper()}  Nx={Nx}  t={ti:.2f}")
        return ln_num, ln_exact

    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    vpath = f"videos/{prefix}.mp4"
    ani.save(vpath, writer=FFMpegWriter(fps=25, bitrate=1800))
    plt.close(fig)
    print(f"  › vídeo {vpath}")
 
# ─── tabla de errores y gráfica de convergencia ─────────────────────────
for norm_lbl, res_dict in [("L2", results_L2), ("L∞", results_Linf)]:
    plt.figure(figsize=(5,4))
    for limiter in limiters:
        Ns, errs = zip(*sorted(res_dict[limiter]))
        h = 1/np.array(Ns)
        ord_glob = -np.polyfit(np.log(h), np.log(errs), 1)[0]
        plt.loglog(h, errs, "o-", label=f"{limiter.upper()}  p≈{ord_glob:.2f}")
    plt.gca().invert_xaxis()
    plt.xlabel("h = 1/Nx"); plt.ylabel(f"Error {norm_lbl}")
    plt.title(f"Convergencia – norma {norm_lbl}")
    plt.grid(ls="--", alpha=.5); plt.legend(); plt.tight_layout()
    plt.savefig(f"convergencia_{norm_lbl}.png", dpi=200)
    plt.close()

print("\n[OK] Resultados y vídeos en carpetas  data/  y  videos/")
