"""Operator splitting methods for the reaction-diffusion equation.

Solves: du/dt = a * d^2u/dx^2 + f(u)
where f(u) = -b*u (linear reaction term)

Demonstrates:
- Forward Euler on full equation
- Ordinary (1st order) splitting
- Strange splitting (1st order)
- Strange splitting (2nd order with Crank-Nicolson and AB2)
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def diffusion_FE(I, a, f, L, dt, F, t, T, step_no, user_action=None):
    """Forward Euler scheme for the diffusion equation.

    Solves: du/dt = a * d^2u/dx^2 + f(u, t)

    Parameters
    ----------
    I : array or callable
        Initial condition (array or function of x)
    a : float
        Diffusion coefficient
    f : callable or None
        Source term f(u, t), or None/0 for no source
    L : float
        Domain length [0, L]
    dt : float
        Time step
    F : float
        Fourier number = a*dt/dx^2
    t : array
        Global time mesh
    T : float
        End time for this solve
    step_no : int
        Starting step number in global time array
    user_action : callable, optional
        Callback function(u, x, t, n)

    Returns
    -------
    u : array
        Solution at final time
    """
    Nt = int(round(T / float(dt)))
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)

    u = np.zeros(Nx + 1)
    u_1 = np.zeros(Nx + 1)

    # Handle source term
    if f is None or f == 0:

        def f(u, t):
            return np.zeros_like(u) if isinstance(u, np.ndarray) else 0

    # Set initial condition
    if isinstance(I, np.ndarray):
        u_1[:] = I
    else:
        for i in range(Nx + 1):
            u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, step_no)

    for n in range(Nt):
        # Interior points: Forward Euler
        u[1:-1] = (
            u_1[1:-1]
            + F * (u_1[:-2] - 2 * u_1[1:-1] + u_1[2:])
            + dt * f(u_1[1:-1], t[step_no + n])
        )
        # Boundary conditions (Dirichlet u=0)
        u[0] = 0
        u[-1] = 0

        if user_action is not None:
            user_action(u, x, t, step_no + n + 1)

        u_1, u = u, u_1

    return u_1


def diffusion_theta(
    I, a, f, L, dt, F, t, T, step_no, theta=0.5, u_L=0, u_R=0, user_action=None
):
    """Theta-rule scheme for the diffusion equation.

    Full solver for the model problem using the theta-rule
    difference approximation in time (no restriction on F,
    i.e., the time step when theta >= 0.5).
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.

    Parameters
    ----------
    I : array or callable
        Initial condition
    a : float
        Diffusion coefficient
    f : callable or None
        Source term f(u, t)
    L : float
        Domain length [0, L]
    dt : float
        Time step
    F : float
        Fourier number = a*dt/dx^2
    t : array
        Global time mesh
    T : float
        End time for this solve
    step_no : int
        Starting step number
    theta : float
        Theta parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit)
    u_L, u_R : float
        Dirichlet boundary values
    user_action : callable, optional
        Callback function(u, x, t, n)

    Returns
    -------
    u : array
        Solution at final time
    """
    Nt = int(round(T / float(dt)))
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros(Nx + 1)
    u_1 = np.zeros(Nx + 1)

    # Build tridiagonal matrix
    diagonal = np.zeros(Nx + 1)
    lower = np.zeros(Nx)
    upper = np.zeros(Nx)
    b = np.zeros(Nx + 1)

    Fl = F * theta
    Fr = F * (1 - theta)
    diagonal[:] = 1 + 2 * Fl
    lower[:] = -Fl
    upper[:] = -Fl
    # Boundary conditions
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0

    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx + 1, Nx + 1),
        format="csr",
    )

    # Handle source term
    if f is None or f == 0:

        def f(u, t):
            return np.zeros_like(u) if isinstance(u, np.ndarray) else 0

    # Set initial condition
    if isinstance(I, np.ndarray):
        u_1[:] = I
    else:
        for i in range(Nx + 1):
            u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, step_no)

    for n in range(Nt):
        b[1:-1] = (
            u_1[1:-1]
            + Fr * (u_1[:-2] - 2 * u_1[1:-1] + u_1[2:])
            + dt * theta * f(u_1[1:-1], t[step_no + n + 1])
            + dt * (1 - theta) * f(u_1[1:-1], t[step_no + n])
        )
        b[0] = u_L
        b[-1] = u_R
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, step_no + n + 1)

        u_1, u = u, u_1

    return u_1


def reaction_FE(I, f, L, Nx, dt, dt_Rfactor, t, step_no, user_action=None):
    """Reaction solver using Forward Euler method.

    Note that t covers the whole global time interval.
    dt is the step of the diffusion part, i.e. there
    is a local time interval [0, dt] the reaction_FE
    deals with each time it is called. step_no keeps
    track of the (global) time step number (required
    for lookup in t).

    Parameters
    ----------
    I : array
        Initial condition (solution from diffusion step)
    f : callable
        Reaction term f(u, t)
    L : float
        Domain length
    Nx : int
        Number of spatial intervals
    dt : float
        Diffusion time step (local interval length)
    dt_Rfactor : int
        Refinement factor for reaction substeps
    t : array
        Global time mesh
    step_no : int
        Current global step number
    user_action : callable, optional
        Callback function

    Returns
    -------
    u : array
        Solution after reaction step
    """
    u = np.copy(I)
    dt_local = dt / float(dt_Rfactor)
    Nt_local = int(round(dt / float(dt_local)))
    x = np.linspace(0, L, Nx + 1)

    for n in range(Nt_local):
        time = t[step_no] + n * dt_local
        u[1:Nx] = u[1:Nx] + dt_local * f(u[1:Nx], time)

    return u


def reaction_AB2(I, f, L, Nx, dt, dt_Rfactor, t, step_no):
    """Reaction solver using 2nd-order Adams-Bashforth method.

    Parameters
    ----------
    I : array
        Initial condition
    f : callable
        Reaction term f(u, t)
    L : float
        Domain length
    Nx : int
        Number of spatial intervals
    dt : float
        Diffusion time step
    dt_Rfactor : int
        Number of substeps for reaction
    t : array
        Global time mesh
    step_no : int
        Current global step number

    Returns
    -------
    u : array
        Solution after reaction step
    """
    u = np.copy(I)
    dt_local = dt / float(dt_Rfactor)
    Nt_local = int(round(dt / float(dt_local)))

    # Store previous f values for AB2
    f_prev = f(u[1:Nx], t[step_no])

    for n in range(Nt_local):
        time = t[step_no] + n * dt_local
        f_curr = f(u[1:Nx], time)

        if n == 0:
            # First step: use Forward Euler
            u[1:Nx] = u[1:Nx] + dt_local * f_curr
        else:
            # AB2: u^{n+1} = u^n + dt/2 * (3*f^n - f^{n-1})
            u[1:Nx] = u[1:Nx] + dt_local * (1.5 * f_curr - 0.5 * f_prev)

        f_prev = f_curr

    return u


def ordinary_splitting(I, a, b, f, L, dt, dt_Rfactor, F, t, T, user_action=None):
    """Ordinary (1st order) operator splitting.

    1st order scheme, i.e. Forward Euler is enough for both
    the diffusion and the reaction part. The time step dt is
    given for the diffusion step, while the time step for the
    reaction part is found as dt/dt_Rfactor, where dt_Rfactor >= 1.
    """
    Nt = int(round(T / float(dt)))
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    u = np.zeros(Nx + 1)

    # Set initial condition
    for i in range(Nx + 1):
        u[i] = I(x[i])

    for n in range(Nt):
        # Step 1: Diffusion
        u_s = diffusion_FE(
            I=u, a=a, f=0, L=L, dt=dt, F=F, t=t, T=dt, step_no=n, user_action=None
        )
        # Step 2: Reaction
        u = reaction_FE(
            I=u_s,
            f=f,
            L=L,
            Nx=Nx,
            dt=dt,
            dt_Rfactor=dt_Rfactor,
            t=t,
            step_no=n,
            user_action=None,
        )

        if user_action is not None:
            user_action(u, x, t, n + 1)


def Strange_splitting_1stOrder(I, a, b, f, L, dt, dt_Rfactor, F, t, T, user_action=None):
    """Strange splitting with Forward Euler (1st order accurate).

    Strange splitting while still using FE for the diffusion
    step and for the reaction step. Gives 1st order scheme.
    Introduce an extra time mesh t2 for the diffusion part,
    since it steps dt/2.
    """
    Nt = int(round(T / float(dt)))
    t2 = np.linspace(0, Nt * dt, (Nt + 1) + Nt)  # Mesh points for half-steps
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    u = np.zeros(Nx + 1)

    # Set initial condition
    for i in range(Nx + 1):
        u[i] = I(x[i])

    for n in range(Nt):
        # Step 1: Half diffusion step
        u_s = diffusion_FE(
            I=u,
            a=a,
            f=0,
            L=L,
            dt=dt / 2.0,
            F=F / 2.0,
            t=t2,
            T=dt / 2.0,
            step_no=2 * n,
            user_action=None,
        )

        # Step 2: Full reaction step
        u_sss = reaction_FE(
            I=u_s,
            f=f,
            L=L,
            Nx=Nx,
            dt=dt,
            dt_Rfactor=dt_Rfactor,
            t=t,
            step_no=n,
            user_action=None,
        )

        # Step 3: Half diffusion step
        u = diffusion_FE(
            I=u_sss,
            a=a,
            f=0,
            L=L,
            dt=dt / 2.0,
            F=F / 2.0,
            t=t2,
            T=dt / 2.0,
            step_no=2 * n + 1,
            user_action=None,
        )

        if user_action is not None:
            user_action(u, x, t, n + 1)


def Strange_splitting_2andOrder(I, a, b, f, L, dt, dt_Rfactor, F, t, T, user_action=None):
    """Strange splitting with Crank-Nicolson and AB2 (2nd order accurate).

    Strange splitting using Crank-Nicolson for the diffusion
    step (theta-rule with theta=0.5) and Adams-Bashforth 2 for
    the reaction step. Gives 2nd order scheme.
    """
    Nt = int(round(T / float(dt)))
    t2 = np.linspace(0, Nt * dt, (Nt + 1) + Nt)  # Mesh points for half-steps
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    u = np.zeros(Nx + 1)

    # Set initial condition
    for i in range(Nx + 1):
        u[i] = I(x[i])

    for n in range(Nt):
        # Step 1: Half diffusion step (Crank-Nicolson)
        u_s = diffusion_theta(
            I=u,
            a=a,
            f=0,
            L=L,
            dt=dt / 2.0,
            F=F / 2.0,
            t=t2,
            T=dt / 2.0,
            step_no=2 * n,
            theta=0.5,
            u_L=0,
            u_R=0,
            user_action=None,
        )

        # Step 2: Full reaction step (AB2)
        u_sss = reaction_AB2(
            I=u_s,
            f=f,
            L=L,
            Nx=Nx,
            dt=dt,
            dt_Rfactor=dt_Rfactor,
            t=t,
            step_no=n,
        )

        # Step 3: Half diffusion step (Crank-Nicolson)
        u = diffusion_theta(
            I=u_sss,
            a=a,
            f=0,
            L=L,
            dt=dt / 2.0,
            F=F / 2.0,
            t=t2,
            T=dt / 2.0,
            step_no=2 * n + 1,
            theta=0.5,
            u_L=0,
            u_R=0,
            user_action=None,
        )

        if user_action is not None:
            user_action(u, x, t, n + 1)


def convergence_rates(scheme="diffusion", Nx_values=None):
    """Compute empirical convergence rates for splitting schemes.

    Parameters
    ----------
    scheme : str
        One of: "diffusion", "ordinary_splitting",
        "Strange_splitting_1stOrder", "Strange_splitting_2andOrder"
    Nx_values : list, optional
        Grid resolutions to test

    Returns
    -------
    dict with E (errors), h (step sizes), r (convergence rates)
    """
    F = 0.5
    T = 1.2
    a = 3.5
    b = 1
    L = 1.5
    k = np.pi / L

    if Nx_values is None:
        Nx_values = [10, 20, 40, 80, 160]

    def exact(x, t):
        """Exact solution to: du/dt = a*d^2u/dx^2 - b*u"""
        return np.exp(-(a * k**2 + b) * t) * np.sin(k * x)

    def f(u, t):
        return -b * u

    def I(x):
        return exact(x, 0)

    E = []
    h = []

    for Nx in Nx_values:
        dx = L / Nx
        dt = F / a * dx**2
        Nt = int(round(T / float(dt)))
        t = np.linspace(0, Nt * dt, Nt + 1)
        x = np.linspace(0, L, Nx + 1)

        # Track maximum error via user_action
        error = [0.0]

        def action(u, x, t_arr, n):
            if n > 0:
                err = np.abs(u - exact(x, t_arr[n])).max()
                error[0] = max(error[0], err)

        if scheme == "diffusion":
            diffusion_FE(I, a, f, L, dt, F, t, T, step_no=0, user_action=action)
        elif scheme == "ordinary_splitting":
            ordinary_splitting(
                I=I,
                a=a,
                b=b,
                f=f,
                L=L,
                dt=dt,
                dt_Rfactor=1,
                F=F,
                t=t,
                T=T,
                user_action=action,
            )
        elif scheme == "Strange_splitting_1stOrder":
            Strange_splitting_1stOrder(
                I=I,
                a=a,
                b=b,
                f=f,
                L=L,
                dt=dt,
                dt_Rfactor=1,
                F=F,
                t=t,
                T=T,
                user_action=action,
            )
        elif scheme == "Strange_splitting_2andOrder":
            Strange_splitting_2andOrder(
                I=I,
                a=a,
                b=b,
                f=f,
                L=L,
                dt=dt,
                dt_Rfactor=1,
                F=F,
                t=t,
                T=T,
                user_action=action,
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        h.append(dt)
        E.append(error[0])

    # Compute convergence rates
    r = [
        np.log(E[i] / E[i - 1]) / np.log(h[i] / h[i - 1])
        for i in range(1, len(Nx_values))
    ]

    return {"E": E, "h": h, "r": r}


def demo():
    """Run convergence rate demonstration for all schemes."""
    schemes = [
        "diffusion",
        "ordinary_splitting",
        "Strange_splitting_1stOrder",
        "Strange_splitting_2andOrder",
    ]

    results = {}
    for scheme in schemes:
        print(f"\nRunning {scheme}...")
        result = convergence_rates(scheme=scheme)
        results[scheme] = result
        print(f"  Errors: {result['E']}")
        print(f"  Rates:  {result['r']}")

    return results


# Run quick convergence test and store result for testing
_test_result = convergence_rates(scheme="diffusion", Nx_values=[10, 20, 40])
RESULT = {
    "errors": _test_result["E"],
    "rates": _test_result["r"],
    "converges": all(0.8 < r < 1.2 for r in _test_result["r"]),  # First-order in dt
}


if __name__ == "__main__":
    demo()
