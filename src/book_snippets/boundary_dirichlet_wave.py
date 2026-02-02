import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# Setup
L, c, T = 1.0, 1.0, 0.2
Nx = 100
C = 0.9  # Courant number
dx = L / Nx
dt = C * dx / c
Nt = int(T / dt)

# Grid and field
grid = Grid(shape=(Nx + 1,), extent=(L,))
t = grid.stepping_dim
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

# Initial condition: plucked string
x_vals = np.linspace(0, L, Nx + 1)
u.data[0, :] = np.sin(np.pi * x_vals)
u.data[1, :] = u.data[0, :]  # Zero initial velocity (demo)

# Equations
update = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.dx2, subdomain=grid.interior)
bc_left = Eq(u[t + 1, 0], 0.0)
bc_right = Eq(u[t + 1, Nx], 0.0)

# Solve
op = Operator([update, bc_left, bc_right])
op(time=Nt, dt=dt)

# Used by tests
RESULT = float(max(abs(u.data[0, 0]), abs(u.data[0, -1])))
