import numpy as np
#import matplotlib.pyplot as plt
import scitools.std as plt

def solver(I, V, m, b, s, F, dt, T, damping='linear'):
    """
    Solve m*u'' + f(u') + s(u) = F(t) for t in (0,T],
    u(0)=I and u'(0)=V,
    by a central finite difference method with time step dt.
    If damping is 'linear', f(u')=b*u, while if damping is
    'quadratic', f(u')=b*u'*abs(u').
    F(t) and s(u) are Python functions.
    """
    dt = float(dt); b = float(b); m = float(m) # avoid integer div.
    Nt = int(round(T/dt))
    u = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)

    u[0] = I
    if damping == 'linear':
        u[1] = u[0] + dt*V + dt**2/(2*m)*(-b*V - s(u[0]) + F(t[0]))
    elif damping == 'quadratic':
        u[1] = u[0] + dt*V + \
               dt**2/(2*m)*(-b*V*abs(V) - s(u[0]) + F(t[0]))

    for n in range(1, Nt):
        if damping == 'linear':
            u[n+1] = (2*m*u[n] + (b*dt/2 - m)*u[n-1] +
                      dt**2*(F(t[n]) - s(u[n])))/(m + b*dt/2)
        elif damping == 'quadratic':
            u[n+1] = (2*m*u[n] - m*u[n-1] + b*u[n]*abs(u[n] - u[n-1])
                      + dt**2*(F(t[n]) - s(u[n])))/\
                      (m + b*abs(u[n] - u[n-1]))
    return u, t

def visualize(u, t, title='', filename='tmp'):
    plt.plot(t, u, 'b-')
    plt.xlabel('t')
    plt.ylabel('u')
    dt = t[1] - t[0]
    plt.title('dt=%g' % dt)
    umin = 1.2*u.min(); umax = 1.2*u.max()
    plt.axis([t[0], t[-1], umin, umax])
    plt.title(title)
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.pdf')
    plt.show()

import sympy as sym

def test_constant():
    """Verify a constant solution."""
    u_exact = lambda t: I
    I = 1.2; V = 0; m = 2; b = 0.9
    w = 1.5
    s = lambda u: w**2*u
    F = lambda t: w**2*u_exact(t)
    dt = 0.2
    T = 2
    u, t = solver(I, V, m, b, s, F, dt, T, 'linear')
    difference = np.abs(u_exact(t) - u).max()
    tol = 1E-13
    assert difference < tol

    u, t = solver(I, V, m, b, s, F, dt, T, 'quadratic')
    difference = np.abs(u_exact(t) - u).max()
    assert difference < tol

def lhs_eq(t, m, b, s, u, damping='linear'):
    """Return lhs of differential equation as sympy expression."""
    v = sym.diff(u, t)
    if damping == 'linear':
        return m*sym.diff(u, t, t) + b*v + s(u)
    else:
        return m*sym.diff(u, t, t) + b*v*sym.Abs(v) + s(u)

def test_quadratic():
    """Verify a quadratic solution."""
    I = 1.2; V = 3; m = 2; b = 0.9
    s = lambda u: 4*u
    t = sym.Symbol('t')
    dt = 0.2
    T = 2

    q = 2  # arbitrary constant
    u_exact = I + V*t + q*t**2
    F = sym.lambdify(t, lhs_eq(t, m, b, s, u_exact, 'linear'))
    u_exact = sym.lambdify(t, u_exact, modules='numpy')
    u1, t1 = solver(I, V, m, b, s, F, dt, T, 'linear')
    diff = np.abs(u_exact(t1) - u1).max()
    tol = 1E-13
    assert diff < tol

    # In the quadratic damping case, u_exact must be linear
    # in order exactly recover this solution
    u_exact = I + V*t
    F = sym.lambdify(t, lhs_eq(t, m, b, s, u_exact, 'quadratic'))
    u_exact = sym.lambdify(t, u_exact, modules='numpy')
    u2, t2 = solver(I, V, m, b, s, F, dt, T, 'quadratic')
    diff = np.abs(u_exact(t2) - u2).max()
    assert diff < tol

def test_sinusoidal():
    """Verify a numerically exact sinusoidal solution when b=F=0."""
    from math import asin

    def u_exact(t):
        w_numerical = 2/dt*np.arcsin(w*dt/2)
        return I*np.cos(w_numerical*t)

    I = 1.2; V = 0; m = 2; b = 0
    w = 1.5  # fix the frequency
    s = lambda u: m*w**2*u
    F = lambda t: 0
    dt = 0.2
    T = 6
    u, t = solver(I, V, m, b, s, F, dt, T, 'linear')
    diff = np.abs(u_exact(t) - u).max()
    tol = 1E-14
    assert diff < tol

    u, t = solver(I, V, m, b, s, F, dt, T, 'quadratic')
    diff = np.abs(u_exact(t) - u).max()
    assert diff < tol

def test_mms():
    """Use method of manufactured solutions."""
    m = 4.; b = 1
    w = 1.5
    t = sym.Symbol('t')
    u_exact = 3*sym.exp(-0.2*t)*sym.cos(1.2*t)
    I = u_exact.subs(t, 0).evalf()
    V = sym.diff(u_exact, t).subs(t, 0).evalf()
    u_exact_py = sym.lambdify(t, u_exact, modules='numpy')
    s = lambda u: u**3
    dt = 0.2
    T = 6
    errors_linear = []
    errors_quadratic = []
    # Run grid refinements and compute exact error
    for i in range(5):
        F_formula = lhs_eq(t, m, b, s, u_exact, 'linear')
        F = sym.lambdify(t, F_formula)
        u1, t1 = solver(I, V, m, b, s, F, dt, T, 'linear')
        error = np.sqrt(np.sum((u_exact_py(t1) - u1)**2)*dt)
        errors_linear.append((dt, error))

        F_formula = lhs_eq(t, m, b, s, u_exact, 'quadratic')
        #print sym.latex(F_formula, mode='plain')
        F = sym.lambdify(t, F_formula)
        u2, t2 = solver(I, V, m, b, s, F, dt, T, 'quadratic')
        error = np.sqrt(np.sum((u_exact_py(t2) - u2)**2)*dt)
        errors_quadratic.append((dt, error))
        dt /= 2
    # Estimate convergence rates
    tol = 0.05
    for errors in errors_linear, errors_quadratic:
        for i in range(1, len(errors)):
            dt, error = errors[i]
            dt_1, error_1 = errors[i-1]
            r = np.log(error/error_1)/np.log(dt/dt_1)
            assert abs(r - 2.0) < tol

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--I', type=float, default=1.0)
    parser.add_argument('--V', type=float, default=0.0)
    parser.add_argument('--m', type=float, default=1.0)
    parser.add_argument('--b', type=float, default=0.0)
    parser.add_argument('--s', type=str, default='u')
    parser.add_argument('--F', type=str, default='0')
    parser.add_argument('--dt', type=float, default=0.05)
    #parser.add_argument('--T', type=float, default=10)
    parser.add_argument('--T', type=float, default=20)
    parser.add_argument('--window_width', type=float, default=30.,
                        help='Number of periods in a window')
    parser.add_argument('--damping', type=str, default='linear')
    parser.add_argument('--savefig', action='store_true')
    # Hack to allow --SCITOOLS options (scitools.std reads this argument
    # at import)
    parser.add_argument('--SCITOOLS_easyviz_backend', default='matplotlib')
    a = parser.parse_args()
    from scitools.std import StringFunction
    s = StringFunction(a.s, independent_variable='u')
    F = StringFunction(a.F, independent_variable='t')
    I, V, m, b, dt, T, window_width, savefig, damping = \
       a.I, a.V, a.m, a.b, a.dt, a.T, a.window_width, a.savefig, \
       a.damping

    u, t = solver(I, V, m, b, s, F, dt, T, damping)

    num_periods = plot_empirical_freq_and_amplitude(u, t)
    if num_periods <= 40:
        plt.figure()
        visualize(u, t)
    else:
        visualize_front(u, t, window_width, savefig)
        visualize_front_ascii(u, t)
    plt.show()

def plot_empirical_freq_and_amplitude(u, t):
    minima, maxima = minmax(t, u)
    p = periods(maxima)
    a = amplitudes(minima, maxima)
    plt.figure()
    from math import pi
    w = 2*pi/p    
    plt.plot(range(len(p)), w, 'r-')
    plt.hold('on')
    plt.plot(range(len(a)), a, 'b-')
    ymax = 1.1*max(w.max(), a.max())
    ymin = 0.9*min(w.min(), a.min())
    plt.axis([0, max(len(p), len(a)), ymin, ymax])
    plt.legend(['estimated frequency', 'estimated amplitude'],
               loc='upper right')
    return len(maxima)

def visualize_front(u, t, window_width, savefig=False):
    """
    Visualize u and the exact solution vs t, using a
    moving plot window and continuous drawing of the
    curves as they evolve in time.
    Makes it easy to plot very long time series.
    P is the approximate duration of one period.
    """
    import scitools.std as st
    from scitools.MovingPlotWindow import MovingPlotWindow

    umin = 1.2*u.min();  umax = -umin
    plot_manager = MovingPlotWindow(
        window_width=window_width,
        dt=t[1]-t[0],
        yaxis=[umin, umax],
        mode='continuous drawing')
    for n in range(1,len(u)):
        if plot_manager.plot(n):
            s = plot_manager.first_index_in_plot
            st.plot(t[s:n+1], u[s:n+1], 'r-1',
                    title='t=%6.3f' % t[n],
                    axis=plot_manager.axis(),
                    show=not savefig) # drop window if savefig
            if savefig:
                print 't=%g' % t[n]
                st.savefig('tmp_vib%04d.png' % n)
        plot_manager.update(n)

def visualize_front_ascii(u, t, fps=10):
    """
    Plot u and the exact solution vs t line by line in a
    terminal window (only using ascii characters).
    Makes it easy to plot very long time series.
    """
    from scitools.avplotter import Plotter
    import time
    umin = 1.2*u.min();  umax = -umin

    p = Plotter(ymin=umin, ymax=umax, width=60, symbols='+o')
    for n in range(len(u)):
        print p.plot(t[n], u[n]), '%.2f' % (t[n])
        time.sleep(1/float(fps))

def minmax(t, u):
    """
    Compute all local minima and maxima of the function u(t),
    represented by discrete points in the arrays u and t.
    Return lists minima and maxima of (t[i],u[i]) extreme points.
    """
    minima = []; maxima = []
    for n in range(1, len(u)-1, 1):
        if u[n-1] > u[n] < u[n+1]:
            minima.append((t[n], u[n]))
        if u[n-1] < u[n] > u[n+1]:
            maxima.append((t[n], u[n]))
    return minima, maxima

def periods(extrema):
    """
    Given a list of (t,u) points of the maxima or minima,
    return an array of the corresponding local periods.
    """
    p = [extrema[n][0] - extrema[n-1][0]
         for n in range(1, len(extrema))]
    return np.array(p)

def amplitudes(minima, maxima):
    """
    Given a list of (t,u) points of the minima and maxima of
    u, return an array of the corresponding local amplitudes.
    """
    # Compare first maxima with first minima and so on
    a = [(abs(maxima[n][1] - minima[n][1]))/2.0
         for n in range(min(len(minima),len(maxima)))]
    return np.array(a)

if __name__ == '__main__':
    main()
    #test_constant()
    #test_sinusoidal()
    #test_mms()
    #test_quadratic()
