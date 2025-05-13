# test_indices_mp5.py – Verificación de ghost cells y MP5 en condiciones periódicas
import numpy as np

# === Módulos del proyecto ===
from config        import NGHOST           # e.g., NGHOST = 2
from utils         import create_mesh_1d, create_U0
from boundary      import apply_bc
from reconstruction import reconstruct

# === Configuración ===
Nx = 8
xmin, xmax = 0.0, 1.0
x_phys, dx = create_mesh_1d(xmin, xmax, Nx)

# === Inicialización de U con valores crecientes (0 a Nx-1) ===
def init(U_phys):
    U_phys[0, :] = np.arange(Nx)  # sólo una variable conservada

U, phys = create_U0(nvars=1, shape_phys=(Nx,), initializer=init)

# === Aplicar condiciones de frontera periódicas ===
apply_bc(U, bc_x=("periodic", "periodic"), reflect_idx=[])

print("\n=== U después de apply_bc (periodic) ===")
print("index :", np.arange(U.shape[1]))
print("U[0]  :", U[0])  # una sola variable conservada

# === Reconstrucción MP5 ===
UL, UR = reconstruct(U, dx, limiter="mp5", axis=None)

print("\n=== UL/UR en TODAS las interfaces (0..Nx) ===")
for i in range(Nx + 1):
    print(f"iface {i:2d}:  UL={UL[0, i]:7.3f}   UR={UR[0, i]:7.3f}")

# === Esquema ASCII de celdas e interfaces ===
print("\n=== Esquema de celdas vs. interfaces ===")
cells = ''.join([f"{j:^7}" for j in range(-NGHOST, Nx + NGHOST)])
faces = ''.join([f"  ^{i:^5}" for i in range(Nx + 1)])
print("cells:", cells)
print("faces:", faces)

# === Aserciones para test de periodicidad ===
# Comprobamos que los extremos coinciden por periodicidad
assert abs(UL[0, 0] - UL[0, Nx]) < 1e-12  # izquierda = derecha (UL)
assert abs(UR[0, 0] - UR[0, Nx]) < 1e-12  # izquierda = derecha (UR)

print("\n✅  tests periódicos superados")

print("""
Leyenda
-------
• Celdas g-1 .. g+Nx   incluyen fantasmas
• faces 0 … Nx  son las interfaces donde UL/UR se pasan al solver
Si las fantasmas (negativas) duplican 5,6,7 (por periodicidad),
y MP5 usa sólo g-2..g+Nx+1, las UL/UR deben ser continuas en
faces 0 y Nx (verifica los números que salieron arriba).
""")
