from devito import Grid, TimeFunction, first_derivative, left

# Create grid and velocity fields
Nx, Ny = 41, 41
Lx, Ly = 2.0, 2.0

grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
x, y = grid.dimensions

u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)
v = TimeFunction(name="v", grid=grid, time_order=1, space_order=2)

# First-order backward differences for advection
# fd_order=1 gives first-order accuracy
# side=left gives backward difference: (u[x] - u[x-dx]) / dx
u_dx = first_derivative(u, dim=x, side=left, fd_order=1)
u_dy = first_derivative(u, dim=y, side=left, fd_order=1)
v_dx = first_derivative(v, dim=x, side=left, fd_order=1)
v_dy = first_derivative(v, dim=y, side=left, fd_order=1)

# Verify the stencil structure
RESULT = {
    "u_dx": str(u_dx),
    "u_dy": str(u_dy),
}
