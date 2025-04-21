import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

xmin, xmax = -0.5, 0.5
# Viscosity
nu = 0.01

# Periodic BC function 
def bc_periodic(u):
    """
    In this case, periodicity is handled exclusively using np.roll 
    in the reconstruction step, so the array is not modified.
    """
    return u


def reconstruction_MUSCL(u, dx):
    """
    MUSCL reconstruction with minmod limiter and periodic BCs.
    Returns two arrays of size Nx+1:
      - ul[i]  = value 'to the left' of interface i
      - ur[i]  = value 'to the right' of interface i
    to be passed to the Riemann solver.
    """
    Nx = u.size

    # 1) Left/right differences (with periodicity via np.roll)
    du_left  = (u - np.roll(u,  1))   # u[i] - u[i-1]
    du_right = (np.roll(u, -1) - u)   # u[i+1] - u[i]
    # 2) Vectorized minmod: slope[i]
    slopes = np.where(du_left * du_right > 0,
                      np.sign(du_left) * np.minimum(np.abs(du_left), np.abs(du_right)),
                      0.0)

    # 3) States at the center of each cell:
    #    uR_cell[i] is the reconstructed value on the right side of cell i
    #    uL_cell[i] is the reconstructed value on the left side of cell i
    uR_cell = u + 0.5 * slopes
    uL_cell = u - 0.5 * slopes

    # 4) Build arrays of size Nx+1 for the interfaces:
    #    at interface i:
    #      ul[i] = uR_cell[i-1 mod Nx]  (value from the left cell)
    #      ur[i] = uL_cell[i   mod Nx]  (value from the right cell)
    idx = np.arange(Nx+1)
    idxL = (idx - 1) % Nx
    idxR = idx       % Nx

    ul = uR_cell[idxL]
    ur = uL_cell[idxR]

    return ul, ur

# Exact Riemann solver for Burgers' equation
def riemann_burgers(ul, ur):
    # ul, ur are arrays of shape (Nx+1,)
    FL = 0.5 * ul**2
    FR = 0.5 * ur**2
    f  = np.zeros_like(ul)

    # Shock → max
    mask_shock = ul > ur
    f[mask_shock] = np.maximum(FL[mask_shock], FR[mask_shock])

    # Rarefaction → min
    mask_rare = ~mask_shock
    f[mask_rare] = np.minimum(FL[mask_rare], FR[mask_rare])

    # Fan that crosses zero → 0
    fan = (ul <= 0) & (ur > 0)
    f[fan] = 0.0

    return f

# Compute dU/dt 
def dudt_burgers(t, u):
    dx = (xmax - xmin) / u.size
    ul, ur = reconstruction_MUSCL(u, dx)    # ul, ur have shape (Nx+1,)
    flux = riemann_burgers(ul, ur)     
    return - (flux[1:] - flux[:-1]) / dx


def dudt_burgers_viscous(t, u):
    dx = (xmax - xmin) / u.size
    ul, ur = reconstruction_MUSCL(u, dx)
    flux = riemann_burgers(ul, ur)

    # convective term
    dudt_conv = - (flux[1:] - flux[:-1]) / dx

    # viscous term
    dudt_visc = np.zeros_like(u)
    dudt_visc[1:-1] = nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

    return dudt_conv + dudt_visc

def dudt_burgers_viscous_conservative(t, u):
    dx = (xmax - xmin) / u.size
    
    # convective
    ul, ur = reconstruction_MUSCL(u, dx)
    flux_conv = riemann_burgers(ul, ur)
    dudt_conv = - (flux_conv[1:] - flux_conv[:-1]) / dx

    #  viscous term in conservative form 
    # q = ∂u/∂x centred differences
    q = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)  # tamaño Nx

    # viscous flux
    flux_visc = nu * q

    # derivative of viscous flux
    dudt_visc = (np.roll(flux_visc, -1) - np.roll(flux_visc, 1)) / (2 * dx)

    return dudt_conv + dudt_visc

# Compute time step via CFL condition
def compute_dt_cfl(u, dx, cfl=0.5):

    max_speed = np.max(np.abs(u))
    
    if max_speed < 1e-10:
        return dx * cfl
    
    dt = cfl * dx / max_speed
    return dt

# RK4 integrator with CFL-based adaptive time step
def RK4(dudt_func, bc_func, t0, q0, tf, n, cfl=0.5):
    """
    RK4 integrator with adaptive time step based on CFL condition.
    """
    Nx = len(q0)
    dx = (xmax - xmin) / Nx
    max_steps = n * 3  # Estimate of maximum steps
    
    q = np.zeros((max_steps, Nx+1))
    q[0, 0] = t0
    q[0, 1:] = q0.copy()
    
    step = 0
    t = t0
    
    while t < tf and step < max_steps - 1:
        step += 1
        u_prev = q[step-1, 1:].copy()
        
        dt_cfl = compute_dt_cfl(u_prev, dx, cfl)
        dt = min(dt_cfl, tf - t)
        
        # No modification imposed here, bc_func is a dummy (returns u)
        u_prev = bc_func(u_prev.copy())
        
        # k1
        k1 = dt * dudt_func(t, u_prev)
        # k2
        u1 = bc_func(u_prev + k1 / 2)
        k2 = dt * dudt_func(t + dt / 2, u1)
        # k3
        u2 = bc_func(u_prev + k2 / 2)
        k3 = dt * dudt_func(t + dt / 2, u2)
        # k4
        u3 = bc_func(u_prev + k3)
        k4 = dt * dudt_func(t + dt, u3)
        
        uf_next = u_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
        q[step, 0] = t
        uf_next = bc_func(uf_next.copy())
        q[step, 1:] = uf_next
    
    return q[:step+1, :]

# MAIN
if __name__ == "__main__":
    # Spatial domain
    Nx = 500
    dx = (xmax - xmin) / Nx
    
    # Cell center grid
    x_centers = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, Nx)
    
    # Initial condition: sine function (periodic in [0,2] since sin(π*0)=sin(2π)=0)
    #u_init = np.sin(np.pi * x_centers)
    u_init = np.exp(-100 * (x_centers)**2)    
    # Time parameters
    t0 = 0.0
    tf = 0.5
    nsteps_estimate = 1000
    cfl = 0.3

    sol = RK4(dudt_burgers_viscous_conservative, bc_periodic, t0, u_init, tf, nsteps_estimate, cfl)
    
    print(f"Simulation completed with {len(sol)} time steps")
    print(f"Final time reached: {sol[-1, 0]:.6f}")
    
    # Extract results
    t_array = sol[:, 0]
    u_array = sol[:, 1:]
    
    dt_values = np.diff(t_array)
    print(f"Minimum time step: {np.min(dt_values):.6f}")
    print(f"Maximum time step: {np.max(dt_values):.6f}")
    print(f"Average time step: {np.mean(dt_values):.6f}")
    
    # Plot the solution as an animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x_centers, u_array[0], lw=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    title = ax.set_title(f'1D Burgers Equation (CFL={cfl})\nt = {t_array[0]:.3f}')

    def update(frame):
        line.set_ydata(u_array[frame])
        title.set_text(f'1D Burgers Equation (CFL={cfl})\nt = {t_array[frame]:.3f}')
        return line,
    
    ani = FuncAnimation(fig, update, frames=len(u_array), interval=1, blit=True)    
    writer = FFMpegWriter(fps=80, codec="libx264", bitrate=-1)
    ani.save("burgers_1d-muscl-nu=0,01-conservative.mp4", writer=writer, dpi=150)

    
    # Compare initial and final solutions
    plt.figure(figsize=(10, 6))
    plt.plot(x_centers, u_array[0], 'b-', label=f't = {t_array[0]:.3f}')
    plt.plot(x_centers, u_array[-1], 'r-', label=f't = {t_array[-1]:.3f}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'1D Burgers Equation - Initial and Final (CFL={cfl})')
    plt.legend()
    plt.savefig("burgers_1d-evol-muscl-nu=0,1-conservative.png")
    
    plt.close('all')
