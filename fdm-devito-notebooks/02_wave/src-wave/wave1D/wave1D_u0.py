#!/usr/bin/env python
"""
1D wave equation with u=0 at the boundary.
Simplest possible implementation.

The key function is::

  u, x, t, cpu = (I, V, f, c, L, dt, C, T, user_action)

which solves the wave equation u_tt = c**2*u_xx on (0,L) with u=0
on x=0,L, for t in (0,T].  Initial conditions: u=I(x), u_t=V(x).

T is the stop time for the simulation.
dt is the desired time step.
C is the Courant number (=c*dt/dx), which specifies dx.
f(x,t) is a function for the source term (can be 0 or None).
I and V are functions of x.

user_action is a function of (u, x, t, n) where the calling
code can add visualization, error computations, etc.
"""
# import sys

# sys.path.insert(1, '../../')

# from ipynb.fs.defs.wave1D_prog import devito_solver
import numpy as np
import time as time
from devito import Constant, Grid, TimeFunction, SparseTimeFunction, Function, Eq, solve, Operator, Buffer

def solver(I, V, f, c, L, dt, C, T, user_action=None):
    """Solve u_tt=c^2*u_xx + f on (0,L)x(0,T]."""
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = dt*c/float(C)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    C2 = C**2                         # Help variable in the scheme
    
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Initialising functions f and V if not provided
    if f is None or f == 0 :
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
        
    t0 = time.perf_counter()  # Measure CPU time

    # Set up grid
    grid = Grid(shape=(Nx+1), extent=(L), dtype=np.float64)
    t_s = grid.stepping_dim
        
    # Create and initialise u
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)
    u.data[0,:] = I(x[:])
    
    if user_action is not None:
        user_action(u.data[0], x, t, 0)

    x_dim = grid.dimensions[0]
    t_dim = grid.time_dim
    
    # The wave equation we are trying to solve
    pde = (1/c**2)*u.dt2-u.dx2
    
    # Source term and injection into equation
    dt_symbolic = grid.time_dim.spacing    
    src = SparseTimeFunction(name='f', grid=grid, npoint=Nx+1, nt=Nt+1)
    
    for i in range(Nt):
        src.data[i] = f(x, t[i])
    
    src.coordinates.data[:, 0] = x
    src_term = src.inject(field=u.forward, expr=src * (dt_symbolic**2))
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Set up special stencil for initial timestep with substitution for u.backward
    v = Function(name='v', grid=grid, npoint=Nx+1, nt=1)
    v.data[:] = V(x[:])
    stencil_init = stencil.subs(u.backward, u.forward - dt_symbolic*v)

    # Boundary conditions
    bc = [Eq(u[t_s+1, 0], 0)]
    bc += [Eq(u[t_s+1, Nx], 0)]

    # Create and apply operators
    op_init = Operator([stencil_init]+src_term+bc)
    op = Operator([stencil]+src_term+bc)
    
    op_init.apply(time_M=1, dt=dt)
    
    if user_action is not None:
        user_action(u.data[1], x, t, 1)
        
    op.apply(time_m=1, dt=dt)
    
    for n in range(2, Nt+1):
        if user_action is not None:
            if user_action(u.data[n], x, t, n):
                break
    
    cpu_time = time.perf_counter() - t0
    
    return u.data[Nt], x, t, cpu_time

def test_quadratic():
    """Check that u(x,t)=x(L-x)(1+t/2) is exactly reproduced."""

    def u_exact(x, t):
        return x*(L-x)*(1 + 0.5*t)

    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0.5*u_exact(x, 0)

    def f(x, t):
        return 2*(1 + 0.5*t)*c**2

    L = 2.5
    c = 1.5
    C = 0.75
    Nx = 6  # Very coarse mesh for this exact test
    dt = C*(L/Nx)/c
    T = 18

    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])
        # Debugging
        print('*********')
        print(n)
        print(u_e)
        print(u)
        print('*********')
        print(np.abs(u- u_e).min())
        diff = np.abs(u - u_e).max()
        tol = 1E-7
        assert diff < tol

    solver(I, V, f, c, L, dt, C, T,
           user_action=assert_no_error)

def test_constant():
    """Check that u(x,t)=Q=0 is exactly reproduced."""
    u_const = 0  # Require 0 because of the boundary conditions
    C = 0.75
    dt = C # Very coarse mesh
    u, x, t, cpu = solver(I=lambda x:
                          0, V=0, f=0, c=1.5, L=2.5,
                          dt=dt, C=C, T=18)
    tol = 1E-14
    assert np.abs(u - u_const).max() < tol

def viz(
    I, V, f, c, L, dt, C, T,  # PDE parameters
    umin, umax,               # Interval for u in plots
    animate=True,             # Simulation with animation?
    tool='matplotlib',        # 'matplotlib' or 'scitools'
    devito_solver_function=solver   # Function with numerical algorithm
    ):
    """Run solver and visualize u at each time level."""

    def plot_u_st(u, x, t, n):
        """user_action function for devito_solver."""
        plt.plot(x, u, 'r-',
                 xlabel='x', ylabel='u',
                 axis=[0, L, umin, umax],
                 title='t=%f' % t[n], show=True)
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig('frame_%04d.png' % n)  # for movie making

    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            """user_action function for devito_solver."""
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u, 'r-')
                plt.xlabel('x');  plt.ylabel('u')
                plt.axis([0, L, umin, umax])
                plt.legend(['t=%f' % t[n]], loc='lower left')
            else:
                self.lines[0].set_ydata(u)
                plt.legend(['t=%f' % t[n]], loc='lower left')
                plt.draw()
            time.sleep(2) if t[n] == 0 else time.sleep(0.2)
            plt.savefig('tmp_%04d.png' % n)  # for movie making

    if tool == 'matplotlib':
        import matplotlib.pyplot as plt
        plot_u = PlotMatplotlib()
    elif tool == 'scitools':
        import scitools.std as plt  # scitools.easyviz interface
        plot_u = plot_u_st
    import time, glob, os

    # Clean up old movie frames
    for filename in glob.glob('tmp_*.png'):
        os.remove(filename)

    # Call devito_solver and do the simulaton
    user_action = plot_u if animate else None
    u, x, t, cpu = devito_solver_function(
        I, V, f, c, L, dt, C, T, user_action)

    # Make video files
    fps = 4  # frames per second
    codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
                     libtheora='ogg')  # video formats
    filespec = 'tmp_%04d.png'
    movie_program = 'ffmpeg'  # or 'avconv'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
              '-vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)

    if tool == 'scitools':
        # Make an HTML play for showing the animation in a browser
        plt.movie('tmp_*.png', encoder='html', fps=fps,
                  output_file='movie.html')
    return cpu

def guitar(C):
    """Triangular wave (pulled guitar string)."""
    L = 0.75
    x0 = 0.8*L
    a = 0.005
    freq = 440
    wavelength = 2*L
    c = freq*wavelength
    omega = 2*pi*freq
    num_periods = 1
    T = 2*pi/omega*num_periods
    # Choose dt the same as the stability limit for Nx=50
    dt = L/50./c

    def I(x):
        return a*x/x0 if x < x0 else a/(L-x0)*(L-x)

    umin = -1.2*a;  umax = -umin
    cpu = viz(I, 0, 0, c, L, dt, C, T, umin, umax,
              animate=True, tool='scitools')


def convergence_rates(
    u_exact,                 # Python function for exact solution
    I, V, f, c, L,           # physical parameters
    dt0, num_meshes, C, T):  # numerical parameters
    """
    Half the time step and estimate convergence rates for
    for num_meshes simulations.
    """
    # First define an appropriate user action function
    global error
    error = 0  # error computed in the user action function

    def compute_error(u, x, t, n):
        global error  # must be global to be altered here
        # (otherwise error is a local variable, different
        # from error defined in the parent function)
        if n == 0:
            error = 0
        else:
            error = max(error, np.abs(u - u_exact(x, t[n])).max())

    # Run finer and finer resolutions and compute true errors
    E = []
    h = []  # dt, devito_solver adjusts dx such that C=dt*c/dx
    dt = dt0
    for i in range(num_meshes):
        solver(I, V, f, c, L, dt, C, T,
               user_action=compute_error)
        # error is computed in the final call to compute_error
        E.append(error)
        h.append(dt)
        dt /= 2  # halve the time step for next simulation
    print('E:')
    print(E)
    print('h:')
    print(h)
    # Convergence rates for two consecutive experiments
    r = [np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
         for i in range(1,num_meshes) if h[i-1] != 0 and h[i] != h[i-1] and E[i-1] != 0]
    return r

def test_convrate_sincos():
    n = m = 2
    L = 1.0
    u_exact = lambda x, t: np.cos(m*np.pi/L*t)*np.sin(m*np.pi/L*x)

    r = convergence_rates(
        u_exact=u_exact,
        I=lambda x: u_exact(x, 0),
        V=lambda x: 0,
        f=0,
        c=1,
        L=L,
        dt0=0.1,
        num_meshes=6,
        C=0.9,
        T=1)
    print('rates sin(x)*cos(t) solution:')
    print([round(r_,2) for r_ in r])
    assert abs(r[-1] - 2) < 0.002

if __name__ == '__main__':
    test_constant()
    test_quadratic()
    import sys
    try:
        C = float(sys.argv[1])
        print('C=%g' % C)
    except IndexError:
        C = 0.85
    print('Courant number: %.2f' % C)
    #guitar(C)
    test_convrate_sincos()