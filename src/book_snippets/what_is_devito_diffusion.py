import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction, solve

# Problem parameters
Nx = 100
L = 1.0
alpha = 1.0  # diffusion coefficient
F = 0.5  # Fourier number (for stability, F <= 0.5)

# Compute dt from stability condition: F = alpha * dt / dx^2
dx = L / Nx
dt = F * dx**2 / alpha

# Create computational grid
grid = Grid(shape=(Nx + 1,), extent=(L,))

# Define the unknown field
u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

# Set initial condition
u.data[0, Nx // 2] = 1.0

# Define the PDE symbolically and solve for u.forward
a = Constant(name="a")
pde = u.dt - a * u.dx2
update = Eq(u.forward, solve(pde, u.forward))

# Create and run the operator
op = Operator([update])
op(time=1000, dt=dt, a=alpha)

# Used by tests
RESULT = float(np.max(u.data[0, :]))
