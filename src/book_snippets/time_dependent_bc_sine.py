import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# Time-dependent Dirichlet boundary condition: u(0,t) = A*sin(omega*t).
# For time-varying BCs, we loop manually and update the boundary each step.
L = 1.0
Nx = 80
c = 1.0
C = 0.9

dx = L / Nx
dt = C * dx / c
Nt = 10

grid = Grid(shape=(Nx + 1,), extent=(L,))
t_dim = grid.stepping_dim
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

u.data[:] = 0.0

# Interior update (wave equation)
update = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.dx2, subdomain=grid.interior)

# Time-independent right BC
bc_right = Eq(u[t_dim + 1, Nx], 0.0)

# Create operator without the time-dependent left BC
op = Operator([update, bc_right])

# Amplitude and frequency
A = 1.0
omega = 2 * np.pi

# Time-stepping loop with manual BC update
for n in range(Nt):
    t_val = n * dt
    # Set time-dependent BC at left boundary
    u.data[(n + 1) % 3, 0] = A * np.sin(omega * t_val)
    op(time=1, dt=dt)

# Check that the left boundary has non-zero values (was driven by sine)
RESULT = float(np.max(np.abs(u.data[:, 0])))
