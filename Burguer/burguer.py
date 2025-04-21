import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Grid and physical parameters
Lx = 1.0
Nx = 200
dx = Lx / Nx
x = np.linspace(0, Lx, Nx, endpoint=False)

# Burgers parameters
nu = 0.0  # viscosity

# Initial condition: sine wave
u0 = np.sin(2 * np.pi * x)
u0_flat = u0.copy()

# Boundary condition: periodic
def apply_periodic(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

# Boundary condition: Dirichlet
def apply_dirichlet(u, value=0.0):
    u[0] = value
    u[-1] = value
    return u

# Spatial derivative for Burgers' equation (central differences)
def dudt_burgers(t, u_flat):
    u = u_flat.copy()
    du = np.zeros_like(u)

    # Compute first derivative (central differences)
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)

    # Compute second derivative (Laplacian)
    d2u_dx2 = np.zeros_like(u)
    d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    # Burgers' equation
    du[1:-1] = -u[1:-1] * du_dx[1:-1] + nu * d2u_dx2[1:-1]

    return du

# RK4 Integrator 
def RK4(dudt_func, bc_func, t0, q0, tf, n):
    dt = (tf - t0) / (n - 1)
    q = np.zeros((n, len(q0) + 1))
    q[0, 0] = t0
    q[0, 1:] = q0

    for i in range(1, n):
        t = q[i - 1, 0]
        u_prev = q[i - 1, 1:].copy()

        # k1
        k1 = dt * dudt_func(t, u_prev)

        # k2
        u1 = bc_func((u_prev + k1 / 2).copy())
        k2 = dt * dudt_func(t + dt / 2, u1)

        # k3
        u2 = bc_func((u_prev + k2 / 2).copy())
        k3 = dt * dudt_func(t + dt / 2, u2)

        # k4
        u3 = bc_func((u_prev + k3).copy())
        k4 = dt * dudt_func(t + dt, u3)

        uf_next = u_prev + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Update time
        q[i, 0] = t + dt

        # Apply boundary conditions
        uf_next = bc_func(uf_next)
        q[i, 1:] = uf_next

    return q

# Time parameters 
CFL = 0.5
dt = CFL * dx / 1.0  # max u ~ 1
tf = 1.0
n_steps = int(tf / dt) + 1

# Boundary condition
#bc_func = apply_dirichlet
bc_func = apply_periodic

# Solve PDE
sol = RK4(dudt_burgers, bc_func, t0=0.0, q0=u0_flat, tf=tf, n=n_steps)
t_array = sol[:, 0]
u_array = sol[:, 1:]

# Animation (x vs u)
fig, ax = plt.subplots()
line, = ax.plot(x, u_array[0], lw=2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title('1D Burgers Equation')

def update(frame):
    line.set_ydata(u_array[frame])
    ax.set_title(f'1D Burgers Equation\nt = {t_array[frame]:.2f}')
    return line,

ani = FuncAnimation(fig, update, frames=n_steps, interval=40)

writer = FFMpegWriter(fps=20, bitrate=1800)
ani.save("burgers_1D.gif", writer=writer)
plt.close(fig)
