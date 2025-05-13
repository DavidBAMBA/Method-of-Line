# equations.py
import numpy as np

# ===============================================================
# >>> 2 D  —  Formulaciones completas
# ===============================================================

class Advection2D:
    """Advección escalar:  U_t + a_x U_x + a_y U_y = 0"""
    def __init__(self, ax=1.0, ay=1.0):
        self.ax, self.ay = ax, ay

    def flux_x(self, U):           return self.ax * U
    def flux_y(self, U):           return self.ay * U
    def max_wave_speed_x(self, U): return abs(self.ax)
    def max_wave_speed_y(self, U): return abs(self.ay)

    def conserved_to_primitive(self, U): return U
    def primitive_to_conserved(self, V): return V


class Burgers2D:
    """Burgers escalar en 2 D"""
    def flux_x(self, U):           return 0.5 * U**2
    def flux_y(self, U):           return 0.5 * U**2
    def max_wave_speed_x(self, U): return np.max(np.abs(U))
    def max_wave_speed_y(self, U): return np.max(np.abs(U))

    def conserved_to_primitive(self, U): return U
    def primitive_to_conserved(self, V): return V


class Euler2D:
    """Euler ideal 2D: U = [rho, rhov_x, rhov_y, E]"""
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    # ---------- flujos ----------
    def flux_x(self, U):
        rho, rhovx, rhovy, E = U
        vx = rhovx / rho
        vy = rhovy / rho
        P = self._pressure(U)
        return np.array([
            rhovx,
            rhovx * vx + P,
            rhovx * vy,
            (E + P) * vx
        ])

    def flux_y(self, U):
        rho, rhovx, rhovy, E = U
        vx = rhovx / rho
        vy = rhovy / rho
        P = self._pressure(U)
        return np.array([
            rhovy,
            rhovy * vx,
            rhovy * vy + P,
            (E + P) * vy
        ])

    # ---------- presión ----------
    def _pressure(self, U, p_floor=1e-10):
        rho, rhovx, rhovy, E = U
        vx = rhovx / rho
        vy = rhovy / rho
        v2 = vx**2 + vy**2
        e = E / rho - 0.5 * v2
        P = (self.gamma - 1.0) * rho * e
        return np.maximum(P, p_floor)

    # ---------- velocidades características ----------
    def max_wave_speed_x(self, U):
        rho, rhovx, _, E = U
        vx = rhovx / rho
        c = np.sqrt(self.gamma * self._pressure(U) / rho)
        return np.abs(vx) + c

    def max_wave_speed_y(self, U):
        rho, _, rhovy, E = U
        vy = rhovy / rho
        c = np.sqrt(self.gamma * self._pressure(U) / rho)
        return np.abs(vy) + c

    # ---------- P ↔ C ----------
    def conserved_to_primitive(self, U):
        rho, rhovx, rhovy, E = U
        vx = rhovx / rho
        vy = rhovy / rho
        P = self._pressure(U)
        return np.array([rho, vx, vy, P])

    def primitive_to_conserved(self, V):
        rho, vx, vy, P = V
        rhovx = rho * vx
        rhovy = rho * vy
        E = P / (self.gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2)
        return np.array([rho, rhovx, rhovy, E])

# ===============================================================
# >>> 1 D  —  Wrappers delgados
#      ‣ El solver 1 D sólo llamará a flux_x / max_wave_speed_x
# ===============================================================

class Advection1D(Advection2D):
    def __init__(self, a=1.0):
        super().__init__(ax=a, ay=0.0)   # ay=0, no se usa

class Burgers1D(Burgers2D):
    pass  # Sin cambios: usa sólo los métodos en x

class Euler1D(Euler2D):
    """Euler 1D: solo rho, rhov, E"""
    
    def flux_y(self, U): raise NotImplementedError
    def max_wave_speed_y(self, U): raise NotImplementedError

    def flux_x(self, U):
        rho, mom, E = U
        v = mom / rho
        P = self._pressure(U)
        return np.array([
            mom,
            mom * v + P,
            v * (E + P)
        ])

    def max_wave_speed_x(self, U):
        rho, mom, E = U
        v = mom / rho
        c = np.sqrt(self.gamma * self._pressure(U) / rho)
        return np.abs(v) + c

    def _pressure(self, U, p_floor=1e-10):
        rho, mom, E = U
        v = mom / rho
        e = E / rho - 0.5 * v**2
        P = (self.gamma - 1.0) * rho * e
        return np.maximum(P, p_floor)


