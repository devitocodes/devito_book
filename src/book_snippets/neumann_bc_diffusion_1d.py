import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# Neumann boundary conditions: du/dx = 0 at both ends for diffusion.
L = 1.0
Nx = 100
alpha = 1.0
F = 0.4  # stable for Forward Euler diffusion in 1D when F <= 0.5

dx = L / Nx
dt = F * dx**2 / alpha
Nt = 25

grid = Grid(shape=(Nx + 1,), extent=(L,))
t = grid.stepping_dim
u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

x = np.linspace(0.0, L, Nx + 1)
u.data[0, :] = np.exp(-((x - 0.5) ** 2) / (2 * 0.05**2))

update = Eq(u.forward, u + alpha * dt * u.dx2, subdomain=grid.interior)

dx_sym = grid.spacing[0]
bc_left = Eq(
    u[t + 1, 0],
    u[t, 0] + alpha * dt * 2.0 * (u[t, 1] - u[t, 0]) / dx_sym**2,
)
bc_right = Eq(
    u[t + 1, Nx],
    u[t, Nx] + alpha * dt * 2.0 * (u[t, Nx - 1] - u[t, Nx]) / dx_sym**2,
)

op = Operator([update, bc_left, bc_right])

for _ in range(Nt):
    op.apply(time_m=0, time_M=0)
    u.data[0, :] = u.data[1, :]

grad_left = float(abs(u.data[0, 1] - u.data[0, 0]))
grad_right = float(abs(u.data[0, -1] - u.data[0, -2]))

RESULT = max(grad_left, grad_right)
