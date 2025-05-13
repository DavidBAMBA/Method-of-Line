# plot_final_advection.py
import numpy as np
import matplotlib.pyplot as plt
import glob, os, re

# === Parámetros básicos (deben coincidir con tu simulación) ================
prefix   = "advection_1d"
limiter  = "weno5"
velocity = 1.0
xmin, xmax = 0.0, 1.0
Nx = 400

# === Leer el último archivo .csv generado ==================================
pat = f"data/{prefix}_{limiter}_*.csv"
csv_files = glob.glob(pat)

def get_step_number_or_mtime(f):
    match = re.search(r"_(\d{5})\.csv$", f)
    if match:
        return int(match.group(1))
    else:
        print(f"[WARN] Nombre no coincide con el patrón esperado: {f}")
        return os.path.getmtime(f)  # Ordena por fecha de modificación

if not csv_files:
    raise FileNotFoundError(f"No se encontraron archivos CSV con patrón: {pat}")

csv_files = sorted(csv_files, key=get_step_number_or_mtime)

last_file = csv_files[-1]

data = np.loadtxt(last_file, delimiter=",", skiprows=1)
x_phys = np.linspace(xmin, xmax, Nx)
u_num = data[:, 1]
t     = data[0, 2]

# === Solución exacta =======================================================
L = xmax - xmin
shift = (x_phys - velocity * t) % L
u_exa = np.exp(-300 * (shift - 0.5)**2)

# === Figura final ==========================================================
plt.figure(figsize=(10,4))
plt.plot(x_phys, u_num, 'o', ms=3, label="Num.")
plt.plot(x_phys, u_exa, '-', lw=2, label="Exacto")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Advección 1-D • {limiter.upper()} • t = {t:.3f}")
plt.legend()
plt.tight_layout()

os.makedirs("videos", exist_ok=True)
png_path = f"videos/{prefix}_{limiter}_final.png"
plt.savefig(png_path, dpi=300)
plt.show()
print(f"[INFO] Figura final guardada en {png_path}")
