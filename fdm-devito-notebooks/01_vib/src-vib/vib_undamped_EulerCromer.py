import numpy as np

def solver(I, w, dt, T):
    """
    Solve v' = - w**2*u, u'=v for t in (0,T], u(0)=I and v(0)=0,
    by an Euler-Cromer method.
    """
    dt = float(dt)
    Nt = int(round(T/dt))
    u = np.zeros(Nt+1)
    v = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)

    v[0] = 0
    u[0] = I
    for n in range(0, Nt):
        v[n+1] = v[n] - dt*w**2*u[n]
        u[n+1] = u[n] + dt*v[n+1]
    return u, v, t

def solver_ic_fix(I, w, dt, T):
    """
    Solve v' = - w**2*u, u'=v for t in (0,T], u(0)=I and v(0)=0,
    by an Euler-Cromer method. Fix the initial condition for
    v such that the scheme becomes fully equivalent to the centered
    scheme for the corresponding 2nd order ODE u'' + u = 0.
    """
    dt = float(dt)
    Nt = int(round(T/dt))
    u = np.zeros(Nt+1)
    v = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)

    v[0] = 0
    u[0] = I
    for n in range(0, Nt):
        if n == 0:
            v[1] = v[0] - 0.5*dt*w**2*u[n]
        else:
            v[n+1] = v[n] - dt*w**2*u[n]
        u[n+1] = u[n] + dt*v[n+1]
    return u, v, t

def solver_adjust_w(I, w, dt, T, adjust_w=True):
    """As solver, but adjust w to fourth order accuracy."""
    dt = float(dt)
    Nt = int(round(T/dt))
    u = np.zeros(Nt+1)
    v = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)
    w_adj = w*(1 - w**2*dt**2/24.) if adjust_w else w

    v[0] = 0
    u[0] = I
    for n in range(0, Nt):
        v[n+1] = v[n] - dt*w_adj**2*u[n]
        u[n+1] = u[n] + dt*v[n+1]
    return u, v, t

def test_solver():
    """
    Test solver with fixed initial condition against
    equivalent scheme for the 2nd-order ODE u'' + u = 0.
    """
    I = 1.2; w = 2.0; T = 5
    dt = 2/w  # longest possible time step
    u, v, t = solver_ic_fix(I, w, dt, T)
    from vib_undamped import solver as solver2  # 2nd-order ODE
    u2, t2 = solver2(I, w, dt, T)
    error = np.abs(u - u2).max()
    tol = 1E-14
    assert error < tol

def demo():
    """
    Demonstrate difference between Euler-Cromer and the
    scheme for the corresponding 2nd-order ODE.
    """
    I = 1.2; w = 2.0; T = 5
    dt = 2/w  # longest possible time step
    from vib_undamped import solver as solver2  # 2nd-order ODE
    import scitools.std as plt
    for k in range(4):
        dt /= 4
        u2, t2 = solver2(I, w, dt, T)
        u, v, t = solver(I, w, dt, T)
        plt.figure()
        plt.plot(t, u, t2, u2,
                 legend=('Euler-Cromer', 'centered scheme for $u''+u=0$'),
                 title='dt=%.3g' % dt)
        raw_input()
        plt.savefig('ECvs2nd_%d' % k + '.png')
        plt.savefig('ECvs2nd_%d' % k + '.pdf')

def convergence_rate():
    """What is the convergence rate of the Euler-Cromer method?"""
    from vib_undamped import convergence_rates
    def solver_wrapper(I, w, dt, T):
        # convergence_rates demands a solver that returns u, t
        u, v, t = solver(I, w, dt, T)
        return u, t

    def solver_ic_fix_wrapper(I, w, dt, T):
        # convergence_rates demands a solver that returns u, t
        u, v, t = solver(I, w, dt, T)
        return u, t

    def solver_adjust_w_wrapper(I, w, dt, T):
        # convergence_rates demands a solver that returns u, t
        u, v, t = solver_adjust_w(I, w, dt, T, True)
        return u, t

    # Plain Euler-Cromer
    r = convergence_rates(8, solver_wrapper)
    print round(r[-1], 1)
    # Does it help to fix the initia condition?
    r = convergence_rates(8, solver_ic_fix_wrapper)
    print round(r[-1], 1)
    # Adjusted w
    r = convergence_rates(8, solver_adjust_w_wrapper)
    print round(r[-1], 1)

if __name__ == '__main__':
    solver(I=3, w=1, dt=0.6283185, T=2)
    #test_solver()
    #demo()
    #convergence_rate()
