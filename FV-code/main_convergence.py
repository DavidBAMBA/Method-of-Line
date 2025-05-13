# main_convergencia_parallel.py
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re, shutil
import multiprocessing as mp

# ── módulos del proyecto ────────────────────────────────────────────────
from equations          import Advection1D
from initial_conditions import gaussian_advection_1d
from reconstruction     import reconstruct
from riemann            import solve_riemann
from solver             import RK4F, dUdt
from utils              import create_mesh_1d, create_U0
from write              import setup_data_folder

# ── parámetros globales ─────────────────────────────────────────────────
xmin, xmax = 0.0, 1.0
a          = 1.0
tf         = 0.3
solver_name = "exact"
limiters    = ["mc", "mp5"]
Ns          = [40, 80, 160, 320]
L           = xmax - xmin

setup_data_folder("data")

_extract_step = lambda f: int(re.search(r"_(\d+)\.csv$", f).group(1)) if re.search(r"_(\d+)\.csv$", f) else -1

def gaussian_exact(x):
    return np.exp(-300 * (x - 0.5)**2)

# ────────────────────────────────────────────────────────────────────────
def run_simulation_parallel(args):
    Nx, limiter = args
    prefix = f"adv_N{Nx}_{limiter}"

    x_phys, dx = create_mesh_1d(xmin, xmax, Nx)
    U0, _      = create_U0(1, (Nx,), initializer=gaussian_advection_1d(x_phys))
    eq         = Advection1D(a=a)

    recon = lambda U, d, axis=None: reconstruct(U, d, limiter=limiter, axis=axis)
    riem  = lambda UL, UR, eq_, axis=None: solve_riemann(UL, UR, eq_, axis,
                                                         solver=solver_name)

    fixed_step = 0.1 * dx * dx

    RK4F(dUdt_func=dUdt,
        t0=0.0, U0=U0, tf=tf,
        dx=dx, dy=None,
        equation=eq,
        reconstruct=recon,
        riemann_solver=riem,
        x=x_phys, y=None,
        bc_x=("periodic", "periodic"),
        fixed_step=fixed_step,
        save_every=1000,
        filename=prefix,
        reconst=limiter)

    csv_files = sorted(glob.glob(f"data/{prefix}_*.csv"), key=_extract_step)
    frames_u, frames_exact, times = [], [], []
    for f in csv_files:
        dat     = np.loadtxt(f, delimiter=",", skiprows=1)
        x_d, u_d, t_real = dat[:, 0], dat[:, 1], dat[0, -1]
        x_shift = np.mod(x_d - a * t_real - xmin, L) + xmin
        u_ex = gaussian_exact(x_shift)

        frames_u.append(u_d)
        frames_exact.append(u_ex)
        times.append(t_real)

    err      = frames_u[-1] - frames_exact[-1]
    err_L2   = np.sqrt(np.mean(err**2))
    err_Linf = np.max(np.abs(err))

    print(f"[DONE] {limiter.upper()}  Nx={Nx}  L2={err_L2:.2e}  L∞={err_Linf:.2e}")
    return (limiter, Nx, err_L2, err_Linf)

# ────────────────────────────────────────────────────────────────────────
def main():
    tasks = [(Nx, lim) for lim in limiters for Nx in Ns]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_simulation_parallel, tasks)

    # Reorganizar resultados
    errors_L2 = {lim: [] for lim in limiters}
    errors_Linf = {lim: [] for lim in limiters}

    for lim, Nx, e2, ei in results:
        errors_L2[lim].append((Nx, e2))
        errors_Linf[lim].append((Nx, ei))

    # Gráfica de convergencia
    fig, axs = plt.subplots(1, 2, figsize=(13,5))
    for ax, (lbl, err_dict) in zip(axs, [("L2", errors_L2), ("L∞", errors_Linf)]):
        for lim in limiters:
            Ns_sorted, errs = zip(*sorted(err_dict[lim]))
            h   = 1.0 / np.array(Ns_sorted)
            lnH = np.log(h); lnE = np.log(errs)
            p   = -np.polyfit(lnH, lnE, 1)[0]

            ax.loglog(h, errs, 'o-', label=f"{lim.upper()}  p≈{p:.2f}")
            print(f"\n[ {lim.upper()} - norma {lbl} ]  orden global ≈ {p:.4f}")
            for i in range(1, len(Ns_sorted)):
                p_loc = (lnE[i-1] - lnE[i]) / (lnH[i-1] - lnH[i])
                print(f"  Nx {Ns_sorted[i-1]:4d}->{Ns_sorted[i]:4d}  p≈{p_loc:.4f}")

        ax.set_xlabel("h = 1/Nx"); ax.set_ylabel(f"Error {lbl}")
        ax.set_title(f"Convergencia ({lbl})")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.invert_xaxis(); ax.legend()

    plt.tight_layout()
    plt.savefig("convergencia_L2_Linf.png", dpi=300)
    plt.show()

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()