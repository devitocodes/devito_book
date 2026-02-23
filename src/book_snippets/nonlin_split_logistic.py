"""Operator splitting methods for the logistic equation.

Demonstrates ordinary splitting, Strange splitting, and exact treatment
of the linear term f_0(u) = u.
"""

import numpy as np


def solver(dt, T, f, f_0, f_1):
    """Solve u'=f by Forward Euler and by splitting: f(u) = f_0(u) + f_1(u).

    Returns solutions from:
    - Forward Euler on full equation
    - Ordinary (1st order) splitting
    - Strange (2nd order) splitting with FE substeps
    - Strange splitting with exact treatment of f_0
    """
    Nt = int(round(T / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)
    u_FE = np.zeros(len(t))
    u_split1 = np.zeros(len(t))  # 1st-order splitting
    u_split2 = np.zeros(len(t))  # 2nd-order splitting
    u_split3 = np.zeros(len(t))  # 2nd-order splitting w/exact f_0

    u_FE[0] = 0.1
    u_split1[0] = 0.1
    u_split2[0] = 0.1
    u_split3[0] = 0.1

    for n in range(len(t) - 1):
        # Forward Euler on full equation
        u_FE[n + 1] = u_FE[n] + dt * f(u_FE[n])

        # Ordinary splitting: f_0 step then f_1 step
        u_s_n = u_split1[n]
        u_s = u_s_n + dt * f_0(u_s_n)
        u_ss_n = u_s
        u_ss = u_ss_n + dt * f_1(u_ss_n)
        u_split1[n + 1] = u_ss

        # Strange splitting: half f_0, full f_1, half f_0
        u_s_n = u_split2[n]
        u_s = u_s_n + dt / 2.0 * f_0(u_s_n)
        u_sss_n = u_s
        u_sss = u_sss_n + dt * f_1(u_sss_n)
        u_ss_n = u_sss
        u_ss = u_ss_n + dt / 2.0 * f_0(u_ss_n)
        u_split2[n + 1] = u_ss

        # Strange splitting with exact f_0 (u' = u has solution u*exp(t))
        u_s_n = u_split3[n]
        u_s = u_s_n * np.exp(dt / 2.0)  # exact
        u_sss_n = u_s
        u_sss = u_sss_n + dt * f_1(u_sss_n)
        u_ss_n = u_sss
        u_ss = u_ss_n * np.exp(dt / 2.0)  # exact
        u_split3[n + 1] = u_ss

    return u_FE, u_split1, u_split2, u_split3, t


# Define the logistic equation terms
def f(u):
    return u * (1 - u)


def f_0(u):
    return u


def f_1(u):
    return -u**2


# Run with dt=0.1 for reasonable accuracy
dt = 0.1
T = 8.0
u_FE, u_split1, u_split2, u_split3, t = solver(dt, T, f, f_0, f_1)

# Exact solution
u_exact = 1 / (1 + 9 * np.exp(-t))

RESULT = {
    "FE_error": float(np.max(np.abs(u_FE - u_exact))),
    "ordinary_split_error": float(np.max(np.abs(u_split1 - u_exact))),
    "strange_split_error": float(np.max(np.abs(u_split2 - u_exact))),
    "strange_exact_error": float(np.max(np.abs(u_split3 - u_exact))),
}
