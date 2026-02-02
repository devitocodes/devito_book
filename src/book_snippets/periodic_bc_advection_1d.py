import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction

# Periodic boundary conditions using copy equations (1D advection).
L = 1.0
Nx = 80
c = 1.0
C = 0.8

dx = L / Nx
dt = C * dx / c

grid = Grid(shape=(Nx + 1,), extent=(L,))
(x_dim,) = grid.dimensions
t = grid.stepping_dim

u = TimeFunction(name="u", grid=grid, time_order=1, space_order=1)

x = np.linspace(0.0, L, Nx + 1)
u.data[0, :] = np.exp(-0.5 * ((x - 0.25) / 0.05) ** 2)
u.data[1, :] = u.data[0, :]

courant = Constant(name="C", value=C)
update = Eq(u.forward, u - courant * (u - u.subs(x_dim, x_dim - x_dim.spacing)))

bc_left = Eq(u[t + 1, 0], u[t, Nx])
bc_right = Eq(u[t + 1, Nx], u[t + 1, 0])

op = Operator([update, bc_left, bc_right])
op.apply(time_m=0, time_M=0, dt=dt)

RESULT = float(abs(u.data[1, 0] - u.data[1, -1]))
