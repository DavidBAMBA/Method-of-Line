import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# Grid and physical parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Advection velocity 
alphax, alphay = 1.0, 1.0

# Initial condition: 2D Gaussian
u0 = np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))
u0_flat = u0.flatten()

# Boundary condition 
def apply_periodic(u):
    return u  # np.roll handles periodicity implicitly

def apply_dirichlet(u, value=0.0):
    u[0, :] = value
    u[-1, :] = value
    u[:, 0] = value
    u[:, -1] = value
    return u

# Spatial derivative: centered differences 
def dudt_centered(t, u_flat):
    u = u_flat.reshape((Nx, Ny))

    du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)

    du = -alphax * du_dx - alphay * du_dy
    return du.flatten()

# RK4 Integrator 
def RK4(dudt_func, bc_func, t0, q0, tf, n):
    dt = (tf - t0) / (n - 1)
    q = np.zeros((n, len(q0) + 1))
    q[0, 0] = t0
    q[0, 1:] = q0
    for i in range(1, n):
        t = q[i - 1, 0]
        u = q[i - 1, 1:].reshape((Nx, Ny))
        u = bc_func(u.copy())               # Apply BCs before each RK step
        u = u.flatten()

        k1 = dt * dudt_func(t, u)
        k2 = dt * dudt_func(t + dt/2, u + k1/2)
        k3 = dt * dudt_func(t + dt/2, u + k2/2)
        k4 = dt * dudt_func(t + dt,   u + k3)

        q[i, 0] = t + dt
        q[i, 1:] = u + (k1 + 2*k2 + 2*k3 + k4)/6
    return q

# Time parameters 
CFL = 0.5
dt = CFL * min(dx / abs(alphax), dy / abs(alphay))
tf = 1.0
n_steps = int(tf / dt) + 1

# Boundary condition 
bc_func = apply_periodic  # or apply_dirichlet

# Solve PDE
sol = RK4(dudt_centered, bc_func, t0=0.0, q0=u0_flat, tf=tf, n=n_steps)
t_array = sol[:, 0]
u_array = sol[:, 1:].reshape((n_steps, Nx, Ny))

# 3D Animation 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y, t)')
ax.set_title('2D Advection (Centered Differences)')

surf = [ax.plot_surface(X, Y, u_array[0], cmap='coolwarm')]

def update(frame):
    surf[0].remove() 
    surf[0] = ax.plot_surface(X, Y, u_array[frame], cmap='coolwarm')
    ax.set_title(f't = {t_array[frame]:.2f}')
    return surf

ani = FuncAnimation(fig, update, frames=n_steps, interval=50)

writer = FFMpegWriter(fps=20, bitrate=1800)
ani.save("advection_2D_centered.mp4", writer=writer)
plt.close(fig) 

