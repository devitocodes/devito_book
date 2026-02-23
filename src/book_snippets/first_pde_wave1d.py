import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# Problem parameters
L = 1.0  # Domain length
c = 1.0  # Wave speed
T = 1.0  # Final time
Nx = 100  # Number of grid points
C = 0.5  # Courant number (for stability)

# Derived parameters
dx = L / Nx
dt = C * dx / c
Nt = int(T / dt)

# Create the computational grid
grid = Grid(shape=(Nx + 1,), extent=(L,))
t_dim = grid.stepping_dim

# Create a time-varying field (2nd order in time, 2nd order in space)
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

# Initial condition: Gaussian pulse
x_coords = np.linspace(0, L, Nx + 1)
x0 = 0.5 * L
sigma = 0.1
u0 = np.exp(-((x_coords - x0) ** 2) / (2 * sigma**2))
u.data[0, :] = u0

# First step for zero initial velocity (second-order accurate)
u_xx_0 = np.zeros_like(u0)
u_xx_0[1:-1] = (u0[2:] - 2 * u0[1:-1] + u0[:-2]) / dx**2
u1 = u0 + 0.5 * dt**2 * c**2 * u_xx_0
u1[0] = 0.0
u1[-1] = 0.0
u.data[1, :] = u1

# Update equation (interior) + Dirichlet boundaries
update = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.dx2, subdomain=grid.interior)
bc_left = Eq(u[t_dim + 1, 0], 0.0)
bc_right = Eq(u[t_dim + 1, Nx], 0.0)

op = Operator([update, bc_left, bc_right])
op(time=Nt, dt=dt)

# Used by tests
RESULT = float(np.max(np.abs(u.data[0, :])))
