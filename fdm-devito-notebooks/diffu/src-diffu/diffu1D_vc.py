"""
Solve the diffusion equation

    u_t = (a(x)*u_x)_x + f(x,t)

on (0,L) with boundary conditions u(0,t) = u_L and u(L,t) = u_R,
for t in (0,T]. Initial condition: u(x,0) = I(x).

The following naming convention of variables are used.

===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
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

``user_action`` is a function of ``(u, x, t, n)``, ``u[i]`` is the
solution at spatial mesh point ``x[i]`` at time ``t[n]``, where the
calling code can add visualization, error computations, data analysis,
store solutions, etc.
"""

import scipy.sparse
import scipy.sparse.linalg
from numpy import linspace, zeros, random, array
import time, sys


def solver(I, a, f, L, Nx, D, T, theta=0.5, u_L=1, u_R=0,
           user_action=None):
    """
    The a variable is an array of length Nx+1 holding the values of
    a(x) at the mesh points.

    Method: (implicit) theta-rule in time.

    Nx is the total number of mesh cells; mesh points are numbered
    from 0 to Nx.
    D = dt/dx**2 and implicitly specifies the time step.
    T is the stop time for the simulation.
    I is a function of x.

    user_action is a function of (u, x, t, n) where the calling code
    can add visualization, error computations, data analysis,
    store solutions, etc.
    """
    import time
    t0 = time.clock()

    x = linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = D*dx**2
    #print 'dt=%g' % dt
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    if isinstance(a, (float,int)):
        a = zeros(Nx+1) + a
    if isinstance(u_L, (float,int)):
        u_L_ = float(u_L)  # must take copy of u_L number
        u_L = lambda t: u_L_
    if isinstance(u_R, (float,int)):
        u_R_ = float(u_R)  # must take copy of u_R number
        u_R = lambda t: u_R_

    u   = zeros(Nx+1)   # solution array at t[n+1]
    u_n = zeros(Nx+1)   # solution at t[n]

    """
    Basic formula in the scheme:

    0.5*(a[i+1] + a[i])*(u[i+1] - u[i]) -
    0.5*(a[i] + a[i-1])*(u[i] - u[i-1])

    0.5*(a[i+1] + a[i])*u[i+1]
    0.5*(a[i] + a[i-1])*u[i-1]
    -0.5*(a[i+1] + 2*a[i] + a[i-1])*u[i]
    """

    Dl = 0.5*D*theta
    Dr = 0.5*D*(1-theta)

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx+1)
    lower    = zeros(Nx)
    upper    = zeros(Nx)
    b        = zeros(Nx+1)

    # Precompute sparse matrix (scipy format)
    diagonal[1:-1] = 1 + Dl*(a[2:] + 2*a[1:-1] + a[:-2])
    lower[:-1] = -Dl*(a[1:-1] + a[:-2])
    upper[1:]  = -Dl*(a[2:] + a[1:-1])
    # Insert boundary conditions
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0

    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx+1, Nx+1),
        format='csr')
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Time loop
    for n in range(0, Nt):
        b[1:-1] = u_n[1:-1] + Dr*(
            (a[2:] + a[1:-1])*(u_n[2:] - u_n[1:-1]) -
            (a[1:-1] + a[0:-2])*(u_n[1:-1] - u_n[:-2])) + \
            dt*theta*f(x[1:-1], t[n+1]) + \
            dt*(1-theta)*f(x[1:-1], t[n])
        # Boundary conditions
        b[0]  = u_L(t[n+1])
        b[-1] = u_R(t[n+1])
        # Solve
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Switch variables before next step
        u_n, u = u, u_n

    t1 = time.clock()
    return t1-t0


def viz(I, a, f, L, Nx, D, T, umin, umax, theta, u_L, u_R,
        animate=True, store_u=False):

    from scitools.std import plot
    solutions = []
    def process_u(u, x, t, n):
        if animate:
            plot(x, u, 'r-', axis=[0, L, umin, umax], title='t=%f' % t[n])
        if t[n] == 0:
            if store_u:
                solutions.append(x)
                solutions.append(t)
                solutions.append(u.copy())
            time.sleep(3)
        else:
            if store_u:
                solutions.append(u.copy())
            #time.sleep(0.1)

    cpu = solver(
        I, a, f, L, Nx, D, T, theta, u_L, u_R, user_action=process_u)
    return cpu, array(solutions)

def fill_a(a_consts, L, Nx):
    """
    *a_consts*: ``[[x0, a0], [x1, a1], ...]`` is a
    piecewise constant function taking the value ``a0`` in ``[x0,x1]``,
    ``a1`` in ``[x1,x2]``, and so forth.

    Return a finite difference function ``a`` on a uniform mesh with
    Nx+1 points in [0, L] where the function takes on the piecewise
    constant values of *a_const*. That is,

    ``a[i] = a_consts[s][1]`` if ``x[i]`` is in subdomain
    ``[a_consts[s][0], a_consts[s+1][0]]``.
    """
    a = zeros(Nx+1)
    x = linspace(0, L, Nx+1)
    s = 0  # subdomain counter
    for i in range(len(x)):
        if s < len(a_consts)-1 and x[i] > a_consts[s+1][0]:
            s += 1
        a[i] = a_consts[s][1]
    return a

def u_exact_stationary(x, a, u_L, u_R):
    """
    Return stationary solution of a 1D variable coefficient
    Laplace equation: (a(x)*v'(x))'=0, v(0)=u_L, v(L)=u_R.

    v(x) = u_L + (u_R-u_L)*(int_0^x 1/a(c)dc / int_0^L 1/a(c)dc)
    """
    Nx = x.size - 1
    g = zeros(Nx+1)    # integral of 1/a from 0 to x
    dx = x[1] - x[0]   # assumed constant
    i = 0
    g[i] = 0.5*dx/a[i]
    for i in range(1, Nx):
        g[i] = g[i-1] + dx/a[i]
    i = Nx
    g[i] = g[i-1] + 0.5*dx/a[i]
    v = u_L + (u_R - u_L)*g/g[-1]
    return v
