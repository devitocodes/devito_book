import sympy as sp

# Symbolic variables
x_sym, t_sym = sp.symbols("x t")
alpha_sym = sp.Symbol("alpha")

# Manufactured solution (arbitrary smooth function)
u_mms = sp.sin(sp.pi * x_sym) * sp.exp(-t_sym)

# Compute required source term: f = u_t - alpha * u_xx
u_t = sp.diff(u_mms, t_sym)
u_xx = sp.diff(u_mms, x_sym, 2)
f_mms = u_t - alpha_sym * u_xx

# Verify the expressions
RESULT = {
    "u_mms": str(u_mms),
    "f_mms": str(sp.simplify(f_mms)),
}
