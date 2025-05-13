# boundary.py
"""
Relleno de ghost-cells para FV con BC periódicas, outflow, reflectivas
y de valor fijo (“fixed”).  La función pública es **apply_bc**.

Convención 1-D
--------------
      g-1 |   g   | … | g+Nx-1 | g+Nx
ghost ←→ física ←→ ghost

Para “fixed” se imponen **también** las celdas físicas g (izq.) y
g+Nx-1 (der.), tal y como exige el tubo de choque de Sod.
"""
import numpy as np
from config import NGHOST

# ──────────────────────────────────────────────────────────────
# utilidades internas
# ──────────────────────────────────────────────────────────────
def _reflect(arr, axis):
    """Devuelve arr volteado en el eje axis."""
    return np.flip(arr, axis=axis)

# ─────────────────────────────────────────────────────────────
# apply_bc : única API pública
# ──────────────────────────────────────────────────────────────
def apply_bc(U, *,
             bc_x=("periodic", "periodic"),
             bc_y=("periodic", "periodic"),
             ng=NGHOST,
             reflect_idx=None,
             fixed_left=None,  fixed_right=None,
             fixed_bottom=None, fixed_top=None):
    """
    Modifica **in-place** las bandas fantasmas (y, para 'fixed', la
    primera/última celda física).

    bc_x / bc_y : ("tipo_izq", "tipo_der"),  tipo ∈
                  {"periodic","outflow","reflect","fixed"}
    reflect_idx : lista de componentes cuyo signo cambia al reflejar.
    fixed_*     : ndarray (nvars,) con el estado conservado deseado.
    """
    if reflect_idx is None:
        reflect_idx = []

    # ─────────────── 1-D ───────────────
    if U.ndim == 2:
        left, right = bc_x
        # ---- izquierda ----
        if left == "periodic":
            U[:, :ng] = U[:, -2*ng:-ng]
        elif left == "outflow":
            U[:, :ng] = U[:, ng:ng+1]
        elif left == "reflect":
            U[:, :ng] = _reflect(U[:, ng:2*ng], axis=1)
            if reflect_idx:
                U[reflect_idx, :ng] *= -1
        elif left == "fixed":
            # ghost + primera física
            U[:, :ng+1] = fixed_left[:, None]
        else:
            raise ValueError(f"BC '{left}' no reconocida (x-left)")

        # ---- derecha ----
        if right == "periodic":
            U[:, -ng:] = U[:, ng:2*ng]
        elif right == "outflow":
            U[:, -ng:] = U[:, -ng-1:-ng]
        elif right == "reflect":
            U[:, -ng:] = _reflect(U[:, -2*ng:-ng], axis=1)
            if reflect_idx:
                U[reflect_idx, -ng:] *= -1
        elif right == "fixed":
            # última física + ghost
            U[:, -ng-1:] = fixed_right[:, None]
        else:
            raise ValueError(f"BC '{right}' no reconocida (x-right)")
        return

    # ─────────────── 2-D ───────────────
    if U.ndim == 3:
        left, right   = bc_x
        bottom, top   = bc_y

        # ---- eje x ----
        if left == "periodic":
            U[:, :ng, :] = U[:, -2*ng:-ng, :]
        elif left == "outflow":
            U[:, :ng, :] = U[:, ng:ng+1, :]
        elif left == "reflect":
            U[:, :ng, :] = _reflect(U[:, ng:2*ng, :], axis=1)
            if reflect_idx:
                U[reflect_idx, :ng, :] *= -1
        elif left == "fixed":
            U[:, :ng+1, :] = fixed_left[:, None, None]
        else:
            raise ValueError(f"BC '{left}' no reconocida (x-left)")

        if right == "periodic":
            U[:, -ng:, :] = U[:, ng:2*ng, :]
        elif right == "outflow":
            U[:, -ng:, :] = U[:, -ng-1:-ng, :]
        elif right == "reflect":
            U[:, -ng:, :] = _reflect(U[:, -2*ng:-ng, :], axis=1)
            if reflect_idx:
                U[reflect_idx, -ng:, :] *= -1
        elif right == "fixed":
            U[:, -ng-1:, :] = fixed_right[:, None, None]
        else:
            raise ValueError(f"BC '{right}' no reconocida (x-right)")

        # ---- eje y ----
        if bottom == "periodic":
            U[:, :, :ng] = U[:, :, -2*ng:-ng]
        elif bottom == "outflow":
            U[:, :, :ng] = U[:, :, ng:ng+1]
        elif bottom == "reflect":
            U[:, :, :ng] = _reflect(U[:, :, ng:2*ng], axis=2)
            if reflect_idx:
                U[reflect_idx, :, :ng] *= -1
        elif bottom == "fixed":
            U[:, :, :ng+1] = fixed_bottom[:, None, None]
        else:
            raise ValueError(f"BC '{bottom}' no reconocida (y-bottom)")

        if top == "periodic":
            U[:, :, -ng:] = U[:, :, ng:2*ng]
        elif top == "outflow":
            U[:, :, -ng:] = U[:, :, -ng-1:-ng]
        elif top == "reflect":
            U[:, :, -ng:] = _reflect(U[:, :, -2*ng:-ng], axis=2)
            if reflect_idx:
                U[reflect_idx, :, -ng:] *= -1
        elif top == "fixed":
            U[:, :, -ng-1:] = fixed_top[:, None, None]
        else:
            raise ValueError(f"BC '{top}' no reconocida (y-top)")
        return

    raise ValueError("U debe ser 2-D (1-D problema) o 3-D (2-D problema).")