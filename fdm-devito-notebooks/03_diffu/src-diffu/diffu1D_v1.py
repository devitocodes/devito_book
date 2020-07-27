#!/usr/bin/env python
"""
Functions for solving a 1D diffusion equations of simplest types
(constant coefficient, no source term):

      u_t = a*u_xx on (0,L)

with boundary conditions u=0 on x=0,L, for t in (0,T].
Initial condition: u(x,0)=I(x).

The following naming convention of variables are used.

===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
F     The dimensionless number a*dt/dx**2, which implicitly
      specifies the time step.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_n   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================

user_action is a function of (u, x, t, n), u[i] is the solution at
spatial mesh point x[i] at time t[n], where the calling code
can add visualization, error computations, data analysis,
store solutions, etc.
"""
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, use_solver

def solver_FE_simple(I, a, L, Nx, F, T):
    """
    Simplest expression of the computational algorithm
    using the Forward Euler method and explicit Python loops.
    For this method F <= 0.5 for stability.
    """
    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    u   = zeros(Nx+1)
    u_n = zeros(Nx+1)

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    for n in range(0, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            u[i] = u_n[i] + F*(u_n[i-1] - 2*u_n[i] + u_n[i+1])

        # Insert boundary conditions
        u[0] = 0;  u[Nx] = 0

        # Switch variables before next step
        u_n, u = u, u_n
    return u


def solver_FE(I, a, L, Nx, F, T,
              user_action=None, version='scalar'):
    """
    Vectorized implementation of solver_FE_simple.
    """
    import time
    t0 = time.clock()

    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    u   = zeros(Nx+1)   # solution array
    u_n = zeros(Nx+1)   # solution at t-dt
    u_2 = zeros(Nx+1)   # solution at t-2*dt

    # Set initial condition
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    for n in range(0, Nt):
        # Update all inner points
        if version == 'scalar':
            for i in range(1, Nx):
                u[i] = u_n[i] + F*(u_n[i-1] - 2*u_n[i] + u_n[i+1])

        elif version == 'vectorized':
            u[1:Nx] = u_n[1:Nx] +  \
                      F*(u_n[0:Nx-1] - 2*u_n[1:Nx] + u_n[2:Nx+1])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        u[0] = 0;  u[Nx] = 0
        if user_action is not None:
            user_action(u, x, t, n+1)

        # Switch variables before next step
        #u_n[:] = u  # slow
        u_n, u = u, u_n

    t1 = time.clock()
    return u, x, t, t1-t0


def solver_BE_simple(I, a, L, Nx, F, T):
    """
    Simplest expression of the computational algorithm
    for the Backward Euler method, using explicit Python loops
    and a dense matrix format for the coefficient matrix.
    """
    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    u   = zeros(Nx+1)
    u_n = zeros(Nx+1)

    # Data structures for the linear system
    A = zeros((Nx+1, Nx+1))
    b = zeros(Nx+1)

    for i in range(1, Nx):
        A[i,i-1] = -F
        A[i,i+1] = -F
        A[i,i] = 1 + 2*F
    A[0,0] = A[Nx,Nx] = 1

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    for n in range(0, Nt):
        # Compute b and solve linear system
        for i in range(1, Nx):
            b[i] = -u_n[i]
        b[0] = b[Nx] = 0
        u[:] = linalg.solve(A, b)

        # Switch variables before next step
        u_n, u = u, u_n
    return u



def solver_BE(I, a, L, Nx, F, T, user_action=None):
    """
    Vectorized implementation of solver_BE_simple using also
    a sparse (tridiagonal) matrix for efficiency.
    """
    import time
    t0 = time.clock()

    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    u   = zeros(Nx+1)   # solution array at t[n+1]
    u_n = zeros(Nx+1)   # solution at t[n]

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx+1)
    upper    = zeros(Nx+1)
    b        = zeros(Nx+1)
    # "Active" values: diagonal[:], upper[1:], lower[:-1]

    # Precompute sparse matrix
    diagonal[:] = 1 + 2*F
    lower[:] = -F  #1
    upper[:] = -F  #1
    # Insert boundary conditions
    diagonal[0] = 1
    diagonal[Nx] = 1
    # Remove unused/inactive values
    upper[0:2] = 0
    lower[-2:] = 0

    diags = [0, -1, 1]
    A = spdiags([diagonal, lower, upper], diags, Nx+1, Nx+1)
    print A.todense()

    # Set initial condition
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    for n in range(0, Nt):
        b = u_n
        b[0] = b[-1] = 0.0  # boundary conditions
        u[:] = spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Switch variables before next step
        u_n, u = u, u_n

    t1 = time.clock()
    return u, x, t, t1-t0


def solver_theta(I, a, L, Nx, F, T, theta=0.5, u_L=0, u_R=0,
                 user_action=None):
    """
    Full solver for the model problem using the theta-rule
    difference approximation in time (no restriction on F,
    i.e., the time step when theta >= 0.5).
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    """
    import time
    t0 = time.clock()

    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    u   = zeros(Nx+1)   # solution array at t[n+1]
    u_n = zeros(Nx+1)   # solution at t[n]

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx+1)
    upper    = zeros(Nx+1)
    b        = zeros(Nx+1)
    # "Active" values: diagonal[:], upper[1:], lower[:-1]

    # Precompute sparse matrix (scipy format)
    diagonal[:] = 1 + 2*Fl
    lower[:] = -Fl  #1
    upper[:] = -Fl  #1
    # Insert boundary conditions
    diagonal[0] = 1
    diagonal[Nx] = 1
    # Remove unused/inactive values
    upper[0:2] = 0
    lower[-2:] = 0

    diags = [0, -1, 1]
    A = spdiags([diagonal, lower, upper], diags, Nx+1, Nx+1)
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Time loop
    for n in range(0, Nt):
        b[1:-1] = u_n[1:-1] + Fr*(u_n[:-2] - 2*u_n[1:-1] + u_n[2:])
        b[0] = u_L; b[-1] = u_R  # boundary conditions
        u[:] = spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Switch variables before next step
        u_n, u = u, u_n

    t1 = time.clock()
    return u, x, t, t1-t0


def viz(I, a, L, Nx, F, T, umin, umax,
        scheme='FE', animate=True):

    def plot_u(u, x, t, n):
        plot(x, u, 'r-', axis=[0, L, umin, umax], title='t=%f' % t[n])
        if t[n] == 0:
            time.sleep(2)
        else:
            time.sleep(0.2)

    user_action = plot_u if animate else lambda u,x,t,n: None

    u, x, t, cpu = eval('solver_'+scheme)\
                   (I, a, L, Nx, F, T,
                    user_action=user_action)
    return u, cpu


def plug(scheme='FE', F=0.5, Nx=50):
    L = 1.
    a = 1
    T = 0.1

    def I(x):
        """Plug profile as initial condition."""
        if abs(x-L/2.0) > 0.1:
            return 0
        else:
            return 1

    u, cpu = viz(I, a, L, Nx, F, T,
                 umin=-0.1, umax=1.1,
                 scheme=scheme, animate=True)
    print 'CPU time:', cpu

    """
    if not allclose(solutions[0], solutions[-1],
                    atol=1.0E-10, rtol=1.0E-12):
        print 'error in computations'
    else:
        print 'correct solution'
    """


def expsin(scheme='FE', F=0.5, m=3):
    L = 10.0
    a = 1
    T = 1.2

    def exact(x, t):
        return exp(-m**2*pi**2*a/L**2*t)*sin(m*pi/L*x)

    def I(x):
        return exact(x, 0)

    Nx = 80
    viz(I, a, L, Nx, F, T, -1, 1, scheme=scheme, animate=True)

    # Convergence study
    def action(u, x, t, n):
        e = abs(u - exact(x, t[n])).max()
        errors.append(e)

    errors = []
    Nx_values = [10, 20, 40, 80, 160]
    for Nx in Nx_values:
        eval('solver_'+scheme)(I, a, L, Nx, F, T, user_action=action)
        dt = F*(L/Nx)**2/a
        print dt, errors[-1]

if __name__ == '__main__':
    import sys, time
    from scitools.std import *

    if len(sys.argv) < 2:
        print """Usage %s function arg1 arg2 arg3 ...""" % sys.argv[0]
        sys.exit(0)
    cmd = '%s(%s)' % (sys.argv[1], ', '.join(sys.argv[2:]))
    print cmd
    eval(cmd)
