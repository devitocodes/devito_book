"""Backward Euler solver for logistic equation with Picard/Newton iteration."""

import numpy as np


def quadratic_roots(a, b, c):
    """Solve ax^2 + bx + c = 0."""
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None
    sqrt_disc = np.sqrt(discriminant)
    return (-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)


def BE_logistic(u0, dt, Nt, choice="Picard", eps_r=1e-3, omega=1, max_iter=1000):
    """Solve logistic equation u' = u(1-u) using Backward Euler.

    Parameters
    ----------
    u0 : float
        Initial condition
    dt : float
        Time step
    Nt : int
        Number of time steps
    choice : str
        Solution method: 'Picard', 'Picard1', 'Newton', 'r1', or 'r2'
    eps_r : float
        Residual tolerance for iteration
    omega : float
        Relaxation parameter (0 < omega <= 1)
    max_iter : int
        Maximum iterations per time step

    Returns
    -------
    u : ndarray
        Solution at all time levels
    iterations : list
        Number of iterations at each time level
    """
    if choice == "Picard1":
        choice = "Picard"
        max_iter = 1

    u = np.zeros(Nt + 1)
    iterations = []
    u[0] = u0

    for n in range(1, Nt + 1):
        a = dt
        b = 1 - dt
        c = -u[n - 1]

        if choice in ("r1", "r2"):
            # Use exact quadratic formula
            r1, r2 = quadratic_roots(a, b, c)
            u[n] = r1 if choice == "r1" else r2
            iterations.append(0)

        elif choice == "Picard":

            def F(u_val):
                return a * u_val**2 + b * u_val + c

            u_ = u[n - 1]
            k = 0
            while abs(F(u_)) > eps_r and k < max_iter:
                u_ = omega * (-c / (a * u_ + b)) + (1 - omega) * u_
                k += 1
            u[n] = u_
            iterations.append(k)

        elif choice == "Newton":

            def F(u_val):
                return a * u_val**2 + b * u_val + c

            def dF(u_val):
                return 2 * a * u_val + b

            u_ = u[n - 1]
            k = 0
            while abs(F(u_)) > eps_r and k < max_iter:
                u_ = u_ - F(u_) / dF(u_)
                k += 1
            u[n] = u_
            iterations.append(k)

    return u, iterations


def CN_logistic(u0, dt, Nt):
    """Solve logistic equation using Crank-Nicolson with geometric mean.

    The geometric mean linearization avoids iteration entirely.
    """
    u = np.zeros(Nt + 1)
    u[0] = u0
    for n in range(0, Nt):
        u[n + 1] = (1 + 0.5 * dt) / (1 + dt * u[n] - 0.5 * dt) * u[n]
    return u


# Test the solvers
dt = 0.1
Nt = 50
u0 = 0.1

u_picard, iters_picard = BE_logistic(u0, dt, Nt, choice="Picard")
u_newton, iters_newton = BE_logistic(u0, dt, Nt, choice="Newton")
u_cn = CN_logistic(u0, dt, Nt)

# Exact solution: u = 1 / (1 + 9*exp(-t))
t = np.linspace(0, Nt * dt, Nt + 1)
u_exact = 1 / (1 + (1 / u0 - 1) * np.exp(-t))

RESULT = {
    "picard_error": float(np.max(np.abs(u_picard - u_exact))),
    "newton_error": float(np.max(np.abs(u_newton - u_exact))),
    "cn_error": float(np.max(np.abs(u_cn - u_exact))),
    "picard_avg_iters": float(np.mean(iters_picard)),
    "newton_avg_iters": float(np.mean(iters_newton)),
}
