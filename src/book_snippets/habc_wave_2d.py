import numpy as np
from devito import Constant, Eq, Grid, Operator, TimeFunction, solve

# 2D wave equation with Hybrid Absorbing Boundary Condition (HABC).
# Combines second-order Higdon ABC with a weighted absorption layer
# for near-optimal reflection reduction with minimal layer width.
Lx, Ly = 2.0, 2.0
Nx, Ny = 80, 80
c = 1.0
CFL = 0.5
pad = 10  # HABC layer width (much thinner than damping)

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

# Interior wave equation with Dirichlet BCs (overwritten by HABC)
c_sq = Constant(name="c_sq")
pde = u.dt2 - c_sq * u.laplace
stencil = Eq(u.forward, solve(pde, u.forward), subdomain=grid.interior)

t_dim = grid.stepping_dim
x_dim, y_dim = grid.dimensions
bc = [
    Eq(u[t_dim + 1, 0, y_dim], 0),
    Eq(u[t_dim + 1, Nx, y_dim], 0),
    Eq(u[t_dim + 1, x_dim, 0], 0),
    Eq(u[t_dim + 1, x_dim, Ny], 0),
]
op = Operator([stencil] + bc)

# HABC weights and Higdon correction
from src.wave.abc_methods import _apply_habc_correction, create_habc_weights

weights = create_habc_weights(pad)

for _n in range(2, Nt + 1):
    op.apply(time_m=1, time_M=1, dt=dt, c_sq=c**2)
    _apply_habc_correction(u.data, Nx, Ny, c, dt, dx, dy, pad, weights)
    u.data[0, :, :] = u.data[1, :, :]
    u.data[1, :, :] = u.data[2, :, :]

RESULT = float(np.max(np.abs(u.data[1, pad:-pad, pad:-pad])))
