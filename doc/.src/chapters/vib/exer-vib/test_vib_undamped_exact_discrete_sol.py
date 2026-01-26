"""Verify exact solution of vib_undamped.solver function."""

import os
import sys

sys.path.insert(0, os.path.join(os.pardir, "src-vib"))
from numpy import abs
from numpy import arcsin as asin
from numpy import cos, pi
from vib_undamped import solver


def test_solver_exact_discrete_solution():
    def tilde_w(w, dt):
        return (2.0 / dt) * asin(w * dt / 2.0)

    def u_numerical_exact(t):
        return I * cos(tilde_w(w, dt) * t)

    w = 2.5
    I = 1.5

    # Estimate period and time step
    P = 2 * pi / w
    num_periods = 4
    T = num_periods * P
    N = 5  # time steps per period
    dt = P / N
    u, t = solver(I, w, dt, T)
    u_e = u_numerical_exact(t)
    error = abs(u_e - u).max()
    # Make a plot in a file, but not on the screen
    import matplotlib.pyplot as plt

    plt.plot(t, u, "bo", label="numerical")
    plt.plot(t, u_e, "r-", label="exact")
    plt.legend()
    plt.savefig("tmp.png")
    plt.close()

    assert error < 1e-14


if __name__ == "__main__":
    test_solver_exact_discrete_solution()
