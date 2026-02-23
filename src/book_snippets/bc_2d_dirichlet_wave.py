import numpy as np
from devito import Eq, Grid, Operator, TimeFunction

# 2D Dirichlet boundary conditions on all edges (wave equation).
Lx = 1.0
Ly = 1.0
Nx = 51
Ny = 51
c = 1.0
C = 0.5

dx = Lx / (Nx - 1)
dt = C * dx / c
Nt = 5

grid = Grid(shape=(Ny, Nx), extent=(Ly, Lx))
t = grid.stepping_dim
x, y = grid.dimensions

u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

xx = np.linspace(0.0, Lx, Nx)
yy = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(xx, yy)

u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.08**2))
u.data[0, :, :] = u0
u.data[1, :, :] = u0  # demo first step

update = Eq(u.forward, 2 * u - u.backward + (c * dt) ** 2 * u.laplace, subdomain=grid.interior)

bc_left = Eq(u[t + 1, x, 0], 0.0)
bc_right = Eq(u[t + 1, x, Ny - 1], 0.0)
bc_bottom = Eq(u[t + 1, 0, y], 0.0)
bc_top = Eq(u[t + 1, Nx - 1, y], 0.0)

op = Operator([update, bc_left, bc_right, bc_bottom, bc_top])
op(time=Nt, dt=dt)

edges = [
    u.data[0, :, 0],
    u.data[0, :, -1],
    u.data[0, 0, :],
    u.data[0, -1, :],
]
RESULT = float(max(np.max(np.abs(e)) for e in edges))
