import numpy as np
import matplotlib.pyplot as plt
from devito import Grid, Eq, solve, TimeFunction, Operator


def solver_FECS(I, U0, v, L, dt, C, T, user_action=None):
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)  # Mesh points in time
    dx = v*dt/C
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)      # Mesh points in space
    
    # Make sure dx and dt are compatible with x and t
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    C = v*dt/dx

    grid = Grid(shape=(Nx+1,), extent=(L,))
    t_s=grid.time_dim

    u = TimeFunction(name='u', grid=grid, space_order=2, save=Nt+1)

    pde = u.dtr + v*u.dxc
    
    stencil = solve(pde, u.forward)
    eq = Eq(u.forward, stencil)
    
    # Set initial condition u(x,0) = I(x)
    u.data[1, :] = [I(xi) for xi in x]
    
    # Insert boundary condition
    bc = [Eq(u[t_s+1, 0], U0)]
    
    op = Operator([eq] + bc)
    op.apply(dt=dt, x_m=1, x_M=Nx-1)
    
    if user_action is not None:
        for n in range(0, Nt + 1):
            user_action(u.data[n], x, t, n)


def solver(I, U0, v, L, dt, C, T, user_action=None,
           scheme='FE', periodic_bc=True):
    Nt = int(round(T/np.float64(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = v*dt/C
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    C = v*dt/dx
    print('dt=%g, dx=%g, Nx=%d, C=%g' % (dt, dx, Nx, C))

    integral = np.zeros(Nt+1)
    
    grid = Grid(shape=(Nx+1,), extent=(L,), dtype=np.float64)

    t_s=grid.time_dim
    
    def u(to=1, so=1):
        u = TimeFunction(name='u', grid=grid, time_order=to, space_order=so, save=Nt+1)
        return u
        
    if scheme == 'FE':
        u   = u(so=2)
        pde = u.dtr + v*u.dxc
        
        pbc = [Eq(u[t_s+1, 0], u[t_s, 0] - 0.5*C*(u[t_s, 1] - u[t_s, Nx]))]
        pbc += [Eq(u[t_s+1, Nx], u[t_s+1, 0])]
        
    elif scheme == 'LF':
        # Use UP scheme for the first timestep
        u = u(to=2, so=2)
        pde0 = u.dtr(fd_order=1) + v*u.dxl(fd_order=1)
        
        stencil0 = solve(pde0, u.forward)
        eq0      = Eq(u.forward, stencil0).subs(t_s, 0)

        pbc0 = [Eq(u[t_s, 0], u[t_s, Nx]).subs(t_s, 0)]
            
        # Now continue with LF scheme
        pde = u.dtc + v*u.dxc
        
        pbc = [Eq(u[t_s+1, 0], u[t_s-1, 0] - C*(u[t_s, 1] - u[t_s, Nx-1]))]
        pbc += [Eq(u[t_s+1, Nx], u[t_s+1, 0])]
    
    elif scheme == 'UP':
        u   = u()
        pde = u.dtr + v*u.dxl
        
        pbc = [Eq(u[t_s, 0], u[t_s, Nx])]
    
    elif scheme == 'LW':
        u = u(so=2)
        pde = u.dtr + v*u.dxc - 0.5*dt*v**2*u.dx2
        
        pbc = [Eq(u[t_s+1, 0], u[t_s, 0] - 0.5*C*(u[t_s, 1] - u[t_s, Nx-1]) + \
                  0.5*C**2*(u[t_s, 1] - 2*u[t_s, 0] + u[t_s, Nx-1]))]
        pbc += [Eq(u[t_s+1, Nx], u[t_s+1, 0])]
    
    else:
        raise ValueError('scheme="%s" not implemented' % scheme)

    stencil = solve(pde, u.forward)
    eq = Eq(u.forward, stencil)
    
    bc_init = [Eq(u[t_s+1, 0], U0).subs(t_s, 0)]
    
    # Set initial condition u(x,0) = I(x)
    u.data[0, :] = [I(xi) for xi in x]
        
    # Compute the integral under the curve
    integral[0] = dx*(0.5*u.data[0][0] + 0.5*u.data[0][Nx] + np.sum(u.data[0][1:Nx]))
    
    if user_action is not None:
        user_action(u.data[0], x, t, 0)

    bc  = [Eq(u[t_s+1, 0], U0)]
    
    if scheme == 'LF':
        op = Operator((pbc0 if periodic_bc else []) + [eq0] + (bc_init if not periodic_bc else []) \
                      + (pbc if periodic_bc else []) + [eq] + (bc if not periodic_bc else []))
    else:       
        op = Operator(bc_init + (pbc if periodic_bc else []) + [eq] + (bc if not periodic_bc else []))
        
    op.apply(dt=dt, x_m=1, x_M=Nx if scheme == 'UP' else Nx-1)

    for n in range(1, Nt+1):
        # Compute the integral under the curve
        integral[n] = dx*(0.5*u.data[n][0] + 0.5*u.data[n][Nx] + np.sum(u.data[n][1:Nx]))

        if user_action is not None:
            user_action(u.data[n], x, t, n)

        print('I:', integral[n])
    return integral


def run_FECS(case):
    """Special function for the FECS case."""
    if case == 'gaussian':
        def I(x):
            return np.exp(-0.5*((x-L/10)/sigma)**2)
    elif case == 'cosinehat':
        def I(x):
            return np.cos(np.pi*5/L*(x - L/10)) if x < L/5 else 0

    L = 1.0
    sigma = 0.02
    legends = []

    def plot(u, x, t, n):
        """Animate and plot every m steps in the same figure."""
        plt.figure(1)
        if n == 0:
            lines = plot(x, u)
        else:
            lines[0].set_ydata(u)
            plt.draw()
            #plt.savefig()
        plt.figure(2)
        m = 40
        if n % m != 0:
            return
        print('t=%g, n=%d, u in [%g, %g] w/%d points' % \
              (t[n], n, u.min(), u.max(), x.size))
        if np.abs(u).max() > 3:  # Instability?
            return
        plt.plot(x, u)
        legends.append('t=%g' % t[n])

    plt.ion()
    U0 = 0
    dt = 0.001
    C = 1
    T = 1
    solver(I=I, U0=U0, v=1.0, L=L, dt=dt, C=C, T=T,
           user_action=plot)
    plt.legend(legends, loc='lower left')
    plt.savefig('tmp.png'); plt.savefig('tmp.pdf')
    plt.axis([0, L, -0.75, 1.1])
    plt.show()


def run(scheme='UP', case='gaussian', C=1, dt=0.01):
    """General admin routine for explicit and implicit solvers."""

    if case == 'gaussian':
        def I(x):
            return np.exp(-0.5*((x-L/10)/sigma)**2)
    elif case == 'cosinehat':
        def I(x):
            return np.cos(np.pi*5/L*(x - L/10)) \
                   if 0 < x < L/5 else 0

    L = 1.0
    sigma = 0.02
    global lines  # needs to be saved between calls to plot

    def plot(u, x, t, n):
        """Plot t=0 and t=0.6 in the same figure."""
        plt.figure(1)
        global lines
        if n == 0:
            lines = plt.plot(x, u)
            plt.axis([x[0], x[-1], -0.5, 1.5])
            plt.xlabel('x'); plt.ylabel('u')
            plt.savefig('tmp_%04d.png' % n)
            plt.savefig('tmp_%04d.pdf' % n)
        else:
            lines[0].set_ydata(u)
            plt.axis([x[0], x[-1], -0.5, 1.5])
            plt.title('C=%g, dt=%g, dx=%g' %
                      (C, t[1]-t[0], x[1]-x[0]))
            plt.legend(['t=%.3f' % t[n]])
            plt.xlabel('x'); plt.ylabel('u')
            plt.draw()
            plt.savefig('tmp_%04d.png' % n)
        plt.figure(2)
        eps = 1E-14
        if abs(t[n] - 0.6) > eps and abs(t[n] - 0) > eps:
            return
        print('t=%g, n=%d, u in [%g, %g] w/%d points' % \
              (t[n], n, u.min(), u.max(), x.size))
        if np.abs(u).max() > 3:  # Instability?
            return
        plt.plot(x, u)
        plt.draw()
        if n > 0:
            y = [I(x_-v*t[n]) for x_ in x]
            plt.plot(x, y, 'k--')
            if abs(t[n] - 0.6) < eps:
                filename = ('tmp_%s_dt%s_C%s' % \
                            (scheme, t[1]-t[0], C)).replace('.', '')
                np.savez(filename, x=x, u=u, u_e=y)

    plt.ion()
    U0 = 0
    T = 0.7
    v = 1
    # Define video formats and libraries
    codecs = dict(flv='flv', mp4='libx264', webm='libvpx',
                  ogg='libtheora')
    # Remove video files
    import glob, os
    for name in glob.glob('tmp_*.png'):
        os.remove(name)
    for ext in codecs:
        name = 'movie.%s' % ext
        if os.path.isfile(name):
            os.remove(name)

    if scheme == 'CN':
        integral = solver_theta(
            I, v, L, dt, C, T, user_action=plot, FE=False)
    elif scheme == 'BE':
        integral = solver_theta(
            I, v, L, dt, C, T, theta=1, user_action=plot)
    else:
        integral = solver(
            I=I, U0=U0, v=v, L=L, dt=dt, C=C, T=T,
            scheme=scheme, user_action=plot)
    # Finish figure(2)
    plt.figure(2)
    plt.axis([0, L, -0.5, 1.1])
    plt.xlabel('$x$');  plt.ylabel('$u$')
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')
    plt.show()
    # Make videos from figure(1) animation files
    for codec in codecs:
        cmd = 'ffmpeg -i tmp_%%04d.png -r 25 -vcodec %s movie.%s' % \
              (codecs[codec], codec)
        os.system(cmd)
    print('Integral of u:', integral.max(), integral.min())


# TODO: IMPLEMENT THIS IN DEVITO
def solver_theta(I, v, L, dt, C, T, theta=0.5, user_action=None, FE=False):
    """
    Full solver for the model problem using the theta-rule
    difference approximation in time (no restriction on F,
    i.e., the time step when theta >= 0.5).
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
    """
    import time;  t0 = time.process_time()  # for measuring the CPU time
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = v*dt/C
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    C = v*dt/dx
    print('dt=%g, dx=%g, Nx=%d, C=%g' % (dt, dx, Nx, C))

    u   = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)
    u_nm1 = np.zeros(Nx+1)
    integral = np.zeros(Nt+1)

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    # Compute the integral under the curve
    integral[0] = dx*(0.5*u_n[0] + 0.5*u_n[Nx] + np.sum(u_n[1:-1]))

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Representation of sparse matrix and right-hand side
    diagonal = np.zeros(Nx+1)
    lower    = np.zeros(Nx)
    upper    = np.zeros(Nx)
    b        = np.zeros(Nx+1)

    # Precompute sparse matrix (scipy format)
    diagonal[:] = 1
    lower[:] = -0.5*theta*C
    upper[:] =  0.5*theta*C
    if FE:
        diagonal[:] += 4./6
        lower[:] += 1./6
        upper[:] += 1./6
    # Insert boundary conditions
    upper[0] = 0
    lower[-1] = 0

    diags = [0, -1, 1]
    import scipy.sparse
    import scipy.sparse.linalg
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
        format='csr')
    #print A.todense()

    # Time loop
    for n in range(0, Nt):
        b[1:-1] = u_n[1:-1] + 0.5*(1-theta)*C*(u_n[:-2] - u_n[2:])
        if FE:
            b[1:-1] += 1./6*u_n[:-2] + 1./6*u_n[:-2] + 4./6*u_n[1:-1]
        b[0] = u_n[Nx]; b[-1] = u_n[0]  # boundary conditions
        b[0] = 0; b[-1] = 0  # boundary conditions
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Compute the integral under the curve
        integral[n+1] = dx*(0.5*u[0] + 0.5*u[Nx] + np.sum(u[1:-1]))

        # Update u_n before next step
        u_n, u = u, u_n

    t1 = time.process_time()
    return integral


if __name__ == '__main__':
    #run(scheme='LF', case='gaussian', C=1)
    #run(scheme='UP', case='gaussian', C=0.8, dt=0.01)
    #run(scheme='LF', case='gaussian', C=0.8, dt=0.001)
    #run(scheme='LF', case='cosinehat', C=0.8, dt=0.01)
    #run(scheme='CN', case='gaussian', C=1, dt=0.01)
    run(scheme='LW', case='gaussian', C=1, dt=0.01)
