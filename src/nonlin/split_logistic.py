"""Operator splitting methods for the logistic equation.

Demonstrates ordinary splitting, Strange splitting, and exact treatment
of the linear term f_0(u) = u.

This module provides both verbose and compact implementations of splitting
methods for educational purposes.
"""

import numpy as np


def exact_solution(t):
    """Exact solution to u' = u(1-u), u(0) = 0.1."""
    return 1 / (1 + 9 * np.exp(-t))


def f(u):
    """Full logistic equation RHS: f(u) = u(1-u)."""
    return u * (1 - u)


def f_0(u):
    """Linear part: f_0(u) = u."""
    return u


def f_1(u):
    """Nonlinear part: f_1(u) = -u^2."""
    return -(u**2)


def solver(dt, T):
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

    # Initial condition
    u_FE[0] = 0.1
    u_split1[0] = 0.1
    u_split2[0] = 0.1
    u_split3[0] = 0.1

    # Ordinary splitting
    for n in range(len(t) - 1):
        # Forward Euler on full equation
        u_FE[n + 1] = u_FE[n] + dt * f(u_FE[n])

        # Ordinary splitting: f_0 step then f_1 step
        u_s = u_split1[n] + dt * f_0(u_split1[n])
        u_split1[n + 1] = u_s + dt * f_1(u_s)

        # Strange splitting: half f_0, full f_1, half f_0
        u_s = u_split2[n] + 0.5 * dt * f_0(u_split2[n])
        u_ss = u_s + dt * f_1(u_s)
        u_split2[n + 1] = u_ss + 0.5 * dt * f_0(u_ss)

        # Strange splitting with exact f_0 (u' = u => u(t) = u_0*exp(t))
        u_s = u_split3[n] * np.exp(0.5 * dt)
        u_ss = u_s + dt * f_1(u_s)
        u_split3[n + 1] = u_ss * np.exp(0.5 * dt)
    # end-splitting-loop

    return u_FE, u_split1, u_split2, u_split3, t


def demo(dt=0.1, T=8.0, plot=False):
    """Run demonstration of splitting methods."""
    u_FE, u_OS, u_SS, u_SS_exact, t = solver(dt, T)
    u_e = exact_solution(t)

    errors = {
        "FE": np.max(np.abs(u_FE - u_e)),
        "ordinary_split": np.max(np.abs(u_OS - u_e)),
        "strange_split": np.max(np.abs(u_SS - u_e)),
        "strange_exact": np.max(np.abs(u_SS_exact - u_e)),
    }

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(t, u_e, "k-", label="exact", linewidth=2)
        plt.plot(t, u_FE, "b--", label="FE")
        plt.plot(t, u_OS, "r-.", label="ordinary split")
        plt.plot(t, u_SS, "g:", label="Strange split")
        plt.plot(t, u_SS_exact, "m-", label="Strange (exact f_0)")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title(f"Splitting methods, dt={dt}")
        plt.savefig("split_logistic.png")
        plt.savefig("split_logistic.pdf")

    return errors


# Run demonstration and store result for testing
_demo_result = demo(dt=0.1, T=8.0)
RESULT = {
    "FE_error": _demo_result["FE"],
    "ordinary_split_error": _demo_result["ordinary_split"],
    "strange_split_error": _demo_result["strange_split"],
    "strange_exact_error": _demo_result["strange_exact"],
}


if __name__ == "__main__":
    print("Logistic equation splitting demonstration")
    print("=" * 50)

    for dt in [0.2, 0.1, 0.05]:
        print(f"\ndt = {dt}:")
        errors = demo(dt=dt)
        for method, err in errors.items():
            print(f"  {method:20s}: max error = {err:.6f}")
