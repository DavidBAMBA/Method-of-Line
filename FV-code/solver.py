# solver.py  ────────────────────────────────────────────────────────────────
import numpy as np
from config   import NGHOST
from boundary import apply_bc
from write    import setup_data_folder, save_all_fields

# ───────────────────────────────────────────────────────────────────────────
# Δt por CFL   (región física nada más)
# ───────────────────────────────────────────────────────────────────────────
def compute_dt_cfl(U, dx, dy, equation, cfl):
    g = NGHOST
    if U.ndim == 2:  # 1D
        max_c = equation.max_wave_speed_x(U[:, g:-g])
        return cfl * dx / np.max(max_c)
    elif U.ndim == 3:  # 2D
        max_cx = equation.max_wave_speed_x(U[:, g:-g, g:-g])
        max_cy = equation.max_wave_speed_y(U[:, g:-g, g:-g])
        max_speed = max(np.max(max_cx), np.max(max_cy))
        return cfl / max_speed * min(dx, dy)
    else:
        raise ValueError("U debe tener 2 (1D) o 3 (2D) dimensiones.")

# ───────────────────────────────────────────────────────────────────────────
# 1)  RHS FV estándar  (2.ᵒ orden global)
# ───────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────
# 1)  RHS FV estándar  (2.ᵒ orden global)
# ───────────────────────────────────────────────────────────────────────────
def dUdt(U, dx, dy, equation,
         reconstruct, riemann_solver):

    g = NGHOST

    # ───────────── 1-D ─────────────
    if U.ndim == 2:
        UL, UR = reconstruct(U, dx, axis=None)
        F      = riemann_solver(UL, UR, equation, axis=None)     # (nvars, Nx+1)
        dFdx   = (F[:, 1:] - F[:, :-1]) / dx

        rhs = np.zeros_like(U)
        rhs[:, g:-g] = -dFdx
        return rhs

    # ───────────── 2-D ─────────────
    elif U.ndim == 3:
        ULx, URx = reconstruct(U, dx, axis=0)
        Fx       = riemann_solver(ULx, URx, equation, axis=0)
        dFdx     = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

        ULy, URy = reconstruct(U, dy, axis=1)
        Fy       = riemann_solver(ULy, URy, equation, axis=1)
        dFdy     = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy

        rhs = np.zeros_like(U)
        rhs[:, g:-g, g:-g] = -(dFdx[:, :, g:-g] + dFdy[:, g:-g, :])
        return rhs

    raise ValueError("U debe tener 2 (1D) o 3 (2D) dimensiones.")


# ───────────────────────────────────────────────────────────────────────────
#  Integrador explícito de Runge–Kutta 4
# ───────────────────────────────────────────────────────────────────────────
def RK4(dUdt_func,
        t0, U0, tf,
        dx, dy,
        equation, reconstruct, riemann_solver,
        x, y,
        bc_x=("periodic", "periodic"),
        bc_y=("periodic", "periodic"),
        cfl=0.5, gamma=1.4,
        max_steps=10000000,
        save_every=10,
        filename="output", reconst="default"):

    t, U = t0, U0.copy()

    # BC inicial
    apply_bc(U, bc_x=bc_x, bc_y=bc_y)

    setup_data_folder("data")
    save_all_fields(U, x, y, step=0, gamma=gamma,
                    prefix=filename, reconstructor=reconst, time=t0)

    step = 0
    while t < tf and step < max_steps:
        dt = min(compute_dt_cfl(U, dx, dy, equation, cfl), tf - t)

        # k1
        k1 = dUdt_func(U, dx, dy, equation,
                       reconstruct, riemann_solver)

        # k2
        U2 = U + 0.5*dt*k1
        apply_bc(U2, bc_x=bc_x, bc_y=bc_y)
        k2 = dUdt_func(U2, dx, dy, equation,
                       reconstruct, riemann_solver)

        # k3
        U3 = U + 0.5*dt*k2
        apply_bc(U3, bc_x=bc_x, bc_y=bc_y)
        k3 = dUdt_func(U3, dx, dy, equation,
                       reconstruct, riemann_solver)

        # k4
        U4 = U + dt*k3
        apply_bc(U4, bc_x=bc_x, bc_y=bc_y)
        k4 = dUdt_func(U4, dx, dy, equation,
                       reconstruct, riemann_solver)

        # update
        U += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        step += 1
        apply_bc(U, bc_x=bc_x, bc_y=bc_y)


        if not np.all(np.isfinite(U)):
            print(f"[ERROR] NaN/Inf en step {step} (t={t:.5f}). Abortando…")
            break

        if step % save_every == 0 or t >= tf:
            print(f"[INFO] step {step:6d}  t = {t:.5f}")
            save_all_fields(U, x, y, step=step, gamma=gamma,
                            prefix=filename, reconstructor=reconst, time=t)

    return


def RK4F(dUdt_func,
        t0, U0, tf,
        dx, dy,
        equation, reconstruct, riemann_solver,
        x, y,
        bc_x=("periodic", "periodic"),
        bc_y=("periodic", "periodic"),
        fixed_step=0.0001, gamma=1.4,
        max_steps=10000000,
        save_every=10,
        filename="output", reconst="default"):

    t, U = t0, U0.copy()

    # BC inicial
    apply_bc(U, bc_x=bc_x, bc_y=bc_y)

    setup_data_folder("data")
    save_all_fields(U, x, y, step=0, gamma=gamma,
                    prefix=filename, reconstructor=reconst, time=t0)

    step = 0
    while t < tf and step < max_steps:
        dt = min(fixed_step, tf - t)

        # k1
        k1 = dUdt_func(U, dx, dy, equation,
                       reconstruct, riemann_solver)

        # k2
        U2 = U + 0.5*dt*k1
        apply_bc(U2, bc_x=bc_x, bc_y=bc_y)
        k2 = dUdt_func(U2, dx, dy, equation,
                       reconstruct, riemann_solver)

        # k3
        U3 = U + 0.5*dt*k2
        apply_bc(U3, bc_x=bc_x, bc_y=bc_y)
        k3 = dUdt_func(U3, dx, dy, equation,
                       reconstruct, riemann_solver)

        # k4
        U4 = U + dt*k3
        apply_bc(U4, bc_x=bc_x, bc_y=bc_y)
        k4 = dUdt_func(U4, dx, dy, equation,
                       reconstruct, riemann_solver)

        # update
        U += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        step += 1
        apply_bc(U, bc_x=bc_x, bc_y=bc_y)


        if not np.all(np.isfinite(U)):
            print(f"[ERROR] NaN/Inf en step {step} (t={t:.5f}). Abortando…")
            break

        if step % save_every == 0 or t >= tf:
            print(f"[INFO] step {step:6d}  t = {t:.5f}")
            save_all_fields(U, x, y, step=step, gamma=gamma,
                            prefix=filename, reconstructor=reconst, time=t)

    return

