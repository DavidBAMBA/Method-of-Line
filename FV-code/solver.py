import numpy as np

def compute_dt_cfl(U, dx, dy, equation, cfl):
    if U.ndim == 2:
        max_c = equation.max_wave_speed_x(U)
        return cfl * dx / max_c
    elif U.ndim == 3:
        max_cx = equation.max_wave_speed_x(U)
        max_cy = equation.max_wave_speed_y(U)
        return cfl * min(dx / max_cx, dy / max_cy)
    else:
        raise ValueError("U debe tener 2 (1D) o 3 (2D) dimensiones.")


def dUdt(U, dx, dy, equation, boundary_func, reconstruct, riemann_solver):
    U = boundary_func(U)  # ← frontera SIEMPRE aplicada
    if U.ndim == 2:
        UL, UR = reconstruct(U, dx, axis=None)
        F = riemann_solver(UL, UR, equation, axis=None)
        dFdx = (F[:, 1:] - F[:, :-1]) / dx
        return -dFdx

    elif U.ndim == 3:
        ULx, URx = reconstruct(U, dx, axis=0)
        Fx = riemann_solver(ULx, URx, equation, axis=0)

        ULy, URy = reconstruct(U, dy, axis=1)
        Fy = riemann_solver(ULy, URy, equation, axis=1)

        dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx
        dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy

        return -(dFdx + dFdy)

    else:
        raise ValueError("U debe tener 2 (1D) o 3 (2D) dimensiones.")


def RK4(dUdt_func, boundary_func,
        t0, U0, tf,
        dx, dy,
        equation, reconstruct, riemann_solver,
        cfl=0.5, max_steps=10000):

    t = t0
    U = U0.copy()
    time_series     = [t]
    solution_series = [U.copy()]

    for step in range(max_steps):
        dt = min(compute_dt_cfl(U, dx, dy, equation, cfl), tf - t)
        if t >= tf:
            break

        # ---------- Etapa 1 ----------
        U1 = boundary_func(U)
        k1 = dUdt_func(U1, dx, dy, equation, boundary_func, reconstruct, riemann_solver)

        # ---------- Etapa 2 ----------
        U2 = boundary_func(U + 0.5 * dt * k1)
        k2 = dUdt_func(U2, dx, dy, equation, boundary_func, reconstruct, riemann_solver)

        # ---------- Etapa 3 ----------
        U3 = boundary_func(U + 0.5 * dt * k2)
        k3 = dUdt_func(U3, dx, dy, equation, boundary_func, reconstruct, riemann_solver)

        # ---------- Etapa 4 ----------
        U4 = boundary_func(U + dt * k3)
        k4 = dUdt_func(U4, dx, dy, equation, boundary_func, reconstruct, riemann_solver)

        # ---------- Actualización final ----------
        U = boundary_func(U + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0))

        t += dt
        time_series.append(t)
        solution_series.append(U.copy())

    return np.array(time_series), np.array(solution_series)
