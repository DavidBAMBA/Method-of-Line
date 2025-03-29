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

# Boundary condition: periodic
def apply_periodic(u):
    u[0, :]   = u[-2, :]  # left border = last interior row
    u[-1, :]  = u[1, :]   # right border = first interior row
    u[:, 0]   = u[:, -2]  # bottom border = last interior column
    u[:, -1]  = u[:, 1]   # top border = first interior column
    return u

# Boundary condition: Dirichlet
def apply_dirichlet(u, value=0.0):
    u[0, :]   = value
    u[-1, :]  = value
    u[:, 0]   = value
    u[:, -1]  = value
    return u

# Spatial derivative: centered differences 
def dudt_centered(t, u_flat):
    u = u_flat.reshape((Nx, Ny))
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)

    # Central differences for interior points
    du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
    du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)

    du_dx[0, :] = 0.0
    du_dx[-1, :] = 0.0
    du_dy[:, 0] = 0.0
    du_dy[:, -1] = 0.0

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

        # Retrieve previous state and flatten it
        u_prev = q[i - 1, 1:].reshape((Nx, Ny))
        uf     = u_prev.flatten()

        # k1
        k1 = dt * dudt_func(t, uf)

        # k2
        u1 = (uf + k1/2).reshape((Nx, Ny))
        u1 = bc_func(u1).flatten()
        k2 = dt * dudt_func(t + dt/2, u1)

        # k3
        u2 = (uf + k2/2).reshape((Nx, Ny))
        u2 = bc_func(u2).flatten()
        k3 = dt * dudt_func(t + dt/2, u2)

        # k4
        u3 = (uf + k3).reshape((Nx, Ny))
        u3 = bc_func(u3).flatten()
        k4 = dt * dudt_func(t + dt, u3)

        # Final RK4 step
        uf_next = uf + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Update time
        q[i, 0] = t + dt

        # Apply boundary
        uf_next_2d = bc_func(uf_next.reshape((Nx, Ny)))
        q[i, 1:] = uf_next_2d.flatten()

    return q

# Time parameters 
CFL = 0.5
dt = CFL * min(dx / abs(alphax), dy / abs(alphay))
tf = 1.0
n_steps = int(tf / dt) + 1

# Boundary condition 
# bc_func = apply_periodic  
bc_func = apply_dirichlet  

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
ani.save("advection_2D_dirichlet.gif", writer=writer)
plt.close(fig) 
