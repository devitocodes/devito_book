import matplotlib.pyplot as plt
from numpy import array, cos, linspace, log, pi, sqrt, zeros
from vib_empirical_analysis import amplitudes, minmax, periods


def solver(I, w, dt, T):
    """
    Solve u'' + w**2*u = 0 for t in (0,T], u(0)=I and u'(0)=0,
    by a central finite difference method with time step dt.
    Use a forward difference for the first time step.
    """
    dt = float(dt)
    Nt = int(round(T / dt))
    u = zeros(Nt + 1)
    t = linspace(0, Nt * dt, Nt + 1)

    u[0] = I
    u[1] = u[0]  # - 0.5*dt**2*w**2*u[0]
    for n in range(1, Nt):
        u[n + 1] = 2 * u[n] - u[n - 1] - dt**2 * w**2 * u[n]
    return u, t


def exact_solution(t, I, w):
    return I * cos(w * t)


def visualize(u, t, I, w):
    plt.plot(t, u, "r--o")
    t_fine = linspace(0, t[-1], 1001)  # very fine mesh for u_e
    u_e = exact_solution(t_fine, I, w)
    plt.plot(t_fine, u_e, "b-")
    plt.legend(["numerical", "exact"], loc="upper left")
    plt.xlabel("t")
    plt.ylabel("u")
    dt = t[1] - t[0]
    plt.title(f"dt={dt:g}")
    umin = 1.2 * u.min()
    umax = -umin
    plt.axis([t[0], t[-1], umin, umax])
    plt.savefig("vib1.png")
    plt.savefig("vib1.pdf")
    plt.savefig("vib1.eps")


def test_three_steps():
    I = 1
    w = 2 * pi
    dt = 0.1
    T = 1
    u_by_hand = array([1.000000000000000, 0.802607911978213, 0.288358920740053])
    u, t = solver(I, w, dt, T)
    difference = abs(u_by_hand - u[:3]).max()
    assert difference < 1e-14, f"Max difference: {difference}"


def convergence_rates(m, num_periods=8):
    """
    Return m-1 empirical estimates of the convergence rate
    based on m simulations, where the time step is halved
    for each simulation.
    """
    w = 0.35
    I = 0.3
    dt = 2 * pi / w / 30  # 30 time step per period 2*pi/w
    T = 2 * pi / w * num_periods
    dt_values = []
    E_values = []
    for _i in range(m):
        u, t = solver(I, w, dt, T)
        u_e = exact_solution(t, I, w)
        E = sqrt(dt * sum((u_e - u) ** 2))
        dt_values.append(dt)
        E_values.append(E)
        dt = dt / 2

    r = [
        log(E_values[i - 1] / E_values[i]) / log(dt_values[i - 1] / dt_values[i])
        for i in range(1, m, 1)
    ]
    return r


def test_convergence_rates():
    r = convergence_rates(m=5, num_periods=8)
    # Accept rate to 1 decimal place
    assert abs(r[-1] - 2.0) < 0.1, f"Convergence rate: {r[-1]}"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--I", type=float, default=1.0)
    parser.add_argument("--w", type=float, default=2 * pi)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--num_periods", type=int, default=5)
    parser.add_argument("--savefig", action="store_true")
    a = parser.parse_args()
    I, w, dt, num_periods, savefig = a.I, a.w, a.dt, a.num_periods, a.savefig
    P = 2 * pi / w  # one period
    T = P * num_periods
    u, t = solver(I, w, dt, T)
    if num_periods <= 10:
        visualize(u, t, I, w)
    else:
        visualize_front(u, t, I, w, savefig)
    # plot_empirical_freq_and_amplitude(u, t, I, w)
    plt.show()


def plot_empirical_freq_and_amplitude(u, t, I, w):
    minima, maxima = minmax(t, u)
    p = periods(maxima)
    a = amplitudes(minima, maxima)
    plt.figure()
    plt.plot(range(len(p)), 2 * pi / p, "r-")
    plt.plot(range(len(a)), a, "b-")
    plt.plot(range(len(p)), [w] * len(p), "r--")
    plt.plot(range(len(a)), [I] * len(a), "b--")
    plt.legend(
        [
            "numerical frequency",
            "numerical amplitude",
            "analytical frequency",
            "anaytical amplitude",
        ],
        loc="center right",
    )


def visualize_front(u, t, I, w, savefig=False, skip_frames=1):
    """
    Visualize u and the exact solution vs t, using a
    moving plot window and continuous drawing of the
    curves as they evolve in time.
    Makes it easy to plot very long time series.
    """
    P = 2 * pi / w  # one period
    window_width = 8 * P
    dt = t[1] - t[0]
    window_points = int(window_width / dt)
    umin = 1.2 * u.min()
    umax = -umin

    plt.ion()
    for n in range(1, len(u)):
        if n % skip_frames != 0:
            continue
        s = max(0, n - window_points)
        plt.clf()
        plt.plot(t[s : n + 1], u[s : n + 1], "r-", label="numerical")
        plt.plot(t[s : n + 1], I * cos(w * t[s : n + 1]), "b-", label="exact")
        plt.title(f"t={t[n]:6.3f}")
        plt.axis([t[s], t[s] + window_width, umin, umax])
        plt.xlabel("t")
        plt.ylabel("u")
        plt.legend()
        if savefig:
            filename = "tmp_vib%04d.png" % n
            plt.savefig(filename)
            print("making plot file", filename, f"at t={t[n]:g}")
        else:
            plt.draw()
            plt.pause(0.001)


if __name__ == "__main__":
    # main()
    r = convergence_rates(6)
    print(r)
