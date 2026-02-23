import numpy as np
from devito import Constant, Eq, Function, Grid, Operator, TimeFunction, solve

# 2D wave equation with sponge (damping) layer absorbing boundaries.
Lx, Ly = 2.0, 2.0
Nx, Ny = 80, 80
c = 1.0
CFL = 0.5
pad = 15  # damping layer width in grid cells

dx, dy = Lx / Nx, Ly / Ny
dt = CFL / (c * np.sqrt(1 / dx**2 + 1 / dy**2))
Nt = int(round(1.0 / dt))
dt = 1.0 / Nt

grid = Grid(shape=(Nx + 1, Ny + 1), extent=(Lx, Ly))
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

# Gaussian initial condition at center
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)
X, Y = np.meshgrid(x, y, indexing="ij")
u.data[0, :, :] = np.exp(-((X - 1.0) ** 2 + (Y - 1.0) ** 2) / (2 * 0.1**2))
u.data[1, :, :] = u.data[0, :, :]

# Build polynomial damping profile: zero in interior, ramps near edges
# sigma_max chosen from theory: 3*c / W where W = pad*dx
sigma_max = 3.0 * c / (pad * dx)
damp = Function(name="damp", grid=grid)
gamma = np.zeros((Nx + 1, Ny + 1))
for i in range(pad):
    d = (pad - i) / pad
    gamma[i, :] = np.maximum(gamma[i, :], sigma_max * d**3)
    gamma[Nx - i, :] = np.maximum(gamma[Nx - i, :], sigma_max * d**3)
for j in range(pad):
    d = (pad - j) / pad
    gamma[:, j] = np.maximum(gamma[:, j], sigma_max * d**3)
    gamma[:, Ny - j] = np.maximum(gamma[:, Ny - j], sigma_max * d**3)
damp.data[:] = gamma

# PDE with damping: u_tt + damp*u_t = c^2 * laplace(u)
c_sq = Constant(name="c_sq")
pde = u.dt2 + damp * u.dt - c_sq * u.laplace
stencil = Eq(u.forward, solve(pde, u.forward))

t_dim = grid.stepping_dim
x_dim, y_dim = grid.dimensions
bc = [
    Eq(u[t_dim + 1, 0, y_dim], 0),
    Eq(u[t_dim + 1, Nx, y_dim], 0),
    Eq(u[t_dim + 1, x_dim, 0], 0),
    Eq(u[t_dim + 1, x_dim, Ny], 0),
]
op = Operator([stencil] + bc)

for _n in range(2, Nt + 1):
    op.apply(time_m=1, time_M=1, dt=dt, c_sq=c**2)
    u.data[0, :, :] = u.data[1, :, :]
    u.data[1, :, :] = u.data[2, :, :]

RESULT = float(np.max(np.abs(u.data[1, pad:-pad, pad:-pad])))
