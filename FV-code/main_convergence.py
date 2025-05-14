# main_convergencia_parallel.py
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re
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
xmin, xmax  = 0.0, 1.0
a           = 1.0
tf          = 1.0
solver_name = "exact"
limiters    = ["mc", "minmod", "weno3", "mp5", "weno5", "wenoz"]
Ns          = [40, 80, 160, 320, 620, 1249]
L           = xmax - xmin

setup_data_folder("data")
os.makedirs("errors", exist_ok=True)

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
    riem  = lambda UL, UR, eq_, axis=None: solve_riemann(UL, UR, eq_, axis, solver=solver_name)

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

    # Leer último CSV generado
    csv_files = sorted(glob.glob(f"data/{prefix}_*.csv"), key=_extract_step)
    if not csv_files:
        print(f"[ERROR] No CSV found for {prefix}")
        return (limiter, Nx, np.nan, np.nan, np.nan)

    data = np.loadtxt(csv_files[-1], delimiter=",", skiprows=1)
    x_d, u_d, t_real = data[:, 0], data[:, 1], data[0, -1]
    x_shift = np.mod(x_d - a * t_real - xmin, L) + xmin
    u_ex = gaussian_exact(x_shift)

    err = u_d - u_ex
    err_L1   = np.mean(np.abs(err))
    err_L2   = np.sqrt(np.mean(err**2))
    err_Linf = np.max(np.abs(err))

    print(f"[DONE] {limiter.upper()}  Nx={Nx}  L1={err_L1:.2e}  L2={err_L2:.2e}  L∞={err_Linf:.2e}")
    return (limiter, Nx, err_L1, err_L2, err_Linf)

# ────────────────────────────────────────────────────────────────────────
def main():
    tasks = [(Nx, lim) for lim in limiters for Nx in Ns]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_simulation_parallel, tasks)

    # Estructuras de errores
    errors_by_limiter = {lim: [] for lim in limiters}

    for lim, Nx, e1, e2, ei in results:
        errors_by_limiter[lim].append((Nx, e1, e2, ei))

    # Guardar errores L1, L2, Linf en CSVs separados
    for i, norm_name in enumerate(["L1", "L2", "Linf"]):
        with open(f"errors/errors_{norm_name}.csv", "w") as f:
            f.write("Limiter,Nx,Error\n")
            for lim in limiters:
                for entry in sorted(errors_by_limiter[lim]):
                    Nx = entry[0]
                    error = entry[i + 1]
                    f.write(f"{lim},{Nx},{error:.8e}\n")

    # Calcular órdenes y guardar tabla
    tabla = []
    print("\n=== Tabla de errores y órdenes por reconstructor ===")
    print(f"{'Limiter':<8} {'Nx':>5} {'L1':>10} {'L2':>10} {'Linf':>10}  {'p_L1':>6} {'p_L2':>6} {'p_Inf':>6}")

    with open("errors/convergencia_ordenes.csv", "w") as fcsv:
        fcsv.write("Limiter,Nx,L1,L2,Linf,p_L1,p_L2,p_Linf\n")
        for lim in limiters:
            rows = sorted(errors_by_limiter[lim])
            prev = None
            for i, (Nx, L1, L2, Linf) in enumerate(rows):
                if prev is None:
                    p1 = p2 = pInf = ""
                else:
                    h1, h2 = 1.0 / prev[0], 1.0 / Nx
                    p1   = (np.log(prev[1]) - np.log(L1)) / (np.log(h1) - np.log(h2))
                    p2   = (np.log(prev[2]) - np.log(L2)) / (np.log(h1) - np.log(h2))
                    pInf = (np.log(prev[3]) - np.log(Linf)) / (np.log(h1) - np.log(h2))
                    p1, p2, pInf = f"{p1:.2f}", f"{p2:.2f}", f"{pInf:.2f}"

                print(f"{lim:<8} {Nx:5d} {L1:10.2e} {L2:10.2e} {Linf:10.2e}  {p1:>6} {p2:>6} {pInf:>6}")
                fcsv.write(f"{lim},{Nx},{L1:.8e},{L2:.8e},{Linf:.8e},{p1},{p2},{pInf}\n")
                tabla.append([lim.upper(), Nx, L1, L2, Linf, p1, p2, pInf])
                prev = (Nx, L1, L2, Linf)

    # Tabla como imagen
    import pandas as pd
    df = pd.DataFrame(tabla,
                      columns=["Limiter", "Nx", "L1", "L2", "Linf", "p_L1", "p_L2", "p_Linf"])
    fig, ax = plt.subplots(figsize=(12, len(df)*0.4))
    ax.axis('off')
    tabla_mpl = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center')
    tabla_mpl.auto_set_font_size(False)
    tabla_mpl.set_fontsize(10)
    tabla_mpl.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig("convergencia_tabla.png", dpi=300)
    plt.close()
    print("[INFO] Tabla guardada en 'convergencia_tabla.png' y 'errors/convergencia_ordenes.csv'")

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
