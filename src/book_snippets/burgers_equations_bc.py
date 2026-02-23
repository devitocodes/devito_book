from devito import (
    Constant,
    Eq,
    Grid,
    Operator,
    TimeFunction,
    first_derivative,
    left,
    solve,
)

# Create grid and velocity fields
Nx, Ny = 41, 41
Lx, Ly = 2.0, 2.0

grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
x, y = grid.dimensions

u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
v = TimeFunction(name="v", grid=grid, time_order=1, space_order=2)

# First-order backward differences for advection
u_dx = first_derivative(u, dim=x, side=left, fd_order=1)
u_dy = first_derivative(u, dim=y, side=left, fd_order=1)
v_dx = first_derivative(v, dim=x, side=left, fd_order=1)
v_dy = first_derivative(v, dim=y, side=left, fd_order=1)

# Viscosity as symbolic constant
nu = Constant(name="nu")

# Burgers equations with backward advection and centered diffusion
# u_t + u*u_x + v*u_y = nu * laplace(u)
eq_u = Eq(u.dt + u * u_dx + v * u_dy, nu * u.laplace, subdomain=grid.interior)
eq_v = Eq(v.dt + u * v_dx + v * v_dy, nu * v.laplace, subdomain=grid.interior)

# Solve for the update expressions
stencil_u = solve(eq_u, u.forward)
stencil_v = solve(eq_v, v.forward)

update_u = Eq(u.forward, stencil_u)
update_v = Eq(v.forward, stencil_v)

# Boundary conditions
t = grid.stepping_dim
bc_value = 1.0  # Boundary condition value

# u boundary conditions
bc_u = [Eq(u[t + 1, 0, y], bc_value)]  # left
bc_u += [Eq(u[t + 1, Nx - 1, y], bc_value)]  # right
bc_u += [Eq(u[t + 1, x, 0], bc_value)]  # bottom
bc_u += [Eq(u[t + 1, x, Ny - 1], bc_value)]  # top

# v boundary conditions (similar)
bc_v = [Eq(v[t + 1, 0, y], bc_value)]  # left
bc_v += [Eq(v[t + 1, Nx - 1, y], bc_value)]  # right
bc_v += [Eq(v[t + 1, x, 0], bc_value)]  # bottom
bc_v += [Eq(v[t + 1, x, Ny - 1], bc_value)]  # top

# Create operator with updates and boundary conditions
op = Operator([update_u, update_v] + bc_u + bc_v)

RESULT = {
    "num_equations": len([update_u, update_v] + bc_u + bc_v),
    "grid_shape": grid.shape,
}
