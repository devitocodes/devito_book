import numpy as np
from devito import Constant, Eq, Function, Grid, Operator, TimeFunction, solve

# 2D wave equation with split-field PML absorbing boundaries.
# Uses separate directional damping (sigma_x, sigma_y) and auxiliary
# fields (phi_x, phi_y) following the Grote-Sim formulation.
Lx, Ly = 2.0, 2.0
Nx, Ny = 80, 80
c = 1.0
CFL = 0.5
pad = 15  # PML width in grid cells

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

# Build directional PML damping profiles
# sigma_max from PML theory: (p+1)*c*ln(1/R)/(2*W)
R_target = 1e-3
order = 3
W = pad * dx
sigma_max = (order + 1) * c * np.log(1.0 / R_target) / (2 * W)

sigma_x_arr = np.zeros((Nx + 1, Ny + 1))
sigma_y_arr = np.zeros((Nx + 1, Ny + 1))
for i in range(pad):
    d = (pad - i) / pad
    sigma_x_arr[i, :] = sigma_max * d**order
    sigma_x_arr[Nx - i, :] = sigma_max * d**order
for j in range(pad):
    d = (pad - j) / pad
    sigma_y_arr[:, j] = sigma_max * d**order
    sigma_y_arr[:, Ny - j] = sigma_max * d**order

sigma_x_fn = Function(name="sigma_x", grid=grid)
sigma_y_fn = Function(name="sigma_y", grid=grid)
sigma_x_fn.data[:] = sigma_x_arr
sigma_y_fn.data[:] = sigma_y_arr

# Auxiliary fields for the split-field PML
phi_x = TimeFunction(name="phi_x", grid=grid, time_order=1, space_order=2)
phi_y = TimeFunction(name="phi_y", grid=grid, time_order=1, space_order=2)

# Grote-Sim PML equation:
# u_tt + (sigma_x + sigma_y)*u_t + sigma_x*sigma_y*u
#   = c^2*laplace(u) + d(phi_x)/dx + d(phi_y)/dy
c_sq = Constant(name="c_sq")
pde = (u.dt2
       + (sigma_x_fn + sigma_y_fn) * u.dt
       + sigma_x_fn * sigma_y_fn * u
       - c_sq * u.laplace
       - phi_x.dx - phi_y.dy)
stencil_u = Eq(u.forward, solve(pde, u.forward))

# Auxiliary field updates (forward Euler):
# phi_x_t = -sigma_x*phi_x + c^2*(sigma_y - sigma_x)*u_x
# phi_y_t = -sigma_y*phi_y + c^2*(sigma_x - sigma_y)*u_y
dt_sym = grid.stepping_dim.spacing
eq_phi_x = Eq(phi_x.forward,
              phi_x + dt_sym * (
                  -sigma_x_fn * phi_x
                  + c_sq * (sigma_y_fn - sigma_x_fn) * u.dx))
eq_phi_y = Eq(phi_y.forward,
              phi_y + dt_sym * (
                  -sigma_y_fn * phi_y
                  + c_sq * (sigma_x_fn - sigma_y_fn) * u.dy))

t_dim = grid.stepping_dim
x_dim, y_dim = grid.dimensions
bc = [
    Eq(u[t_dim + 1, 0, y_dim], 0),
    Eq(u[t_dim + 1, Nx, y_dim], 0),
    Eq(u[t_dim + 1, x_dim, 0], 0),
    Eq(u[t_dim + 1, x_dim, Ny], 0),
]
op = Operator([stencil_u, eq_phi_x, eq_phi_y] + bc)

for _n in range(2, Nt + 1):
    op.apply(time_m=1, time_M=1, dt=dt, c_sq=c**2)
    u.data[0, :, :] = u.data[1, :, :]
    u.data[1, :, :] = u.data[2, :, :]
    phi_x.data[1, :, :] = phi_x.data[0, :, :]
    phi_y.data[1, :, :] = phi_y.data[0, :, :]

RESULT = float(np.max(np.abs(u.data[1, pad:-pad, pad:-pad])))
