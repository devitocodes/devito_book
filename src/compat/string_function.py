"""
StringFunction: A replacement for scitools.std.StringFunction.

Parses mathematical string expressions and converts them to callable functions.
Uses sympy for robust parsing and numpy for vectorized evaluation.
"""

import numpy as np
import sympy as sp


class StringFunction:
    """
    A callable class that evaluates mathematical string expressions.

    Parameters
    ----------
    expression : str
        A mathematical expression as a string, e.g., 'sin(x)*exp(-x**2)'
    independent_variable : str, optional
        The name of the independent variable (default: 'x')
    globals : dict, optional
        Additional global variables/functions to make available during evaluation

    Examples
    --------
    >>> f = StringFunction('sin(x)*exp(-x**2)')
    >>> f(0.5)
    0.3581...

    >>> f = StringFunction('A*sin(w*t)', independent_variable='t')
    >>> f.set_parameters(A=2, w=3.14)
    >>> f(1.0)
    ...
    """

    def __init__(self, expression, independent_variable="x", globals=None):
        self.expression_str = expression
        self.independent_variable = independent_variable
        self._globals = globals or {}
        self._parameters = {}
        self._vectorized = False

        # Parse the expression to find all symbols
        self._parse_expression()

    def _parse_expression(self):
        """Parse the expression and create a callable."""
        # Create sympy symbol for independent variable
        self._indep_sym = sp.Symbol(self.independent_variable)

        # Try to parse with sympy for validation and to find symbols
        try:
            # Add common mathematical functions to local namespace
            local_dict = {
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "pi": sp.pi,
                "e": sp.E,
                "abs": sp.Abs,
                "sinh": sp.sinh,
                "cosh": sp.cosh,
                "tanh": sp.tanh,
                "asin": sp.asin,
                "acos": sp.acos,
                "atan": sp.atan,
                self.independent_variable: self._indep_sym,
            }

            # Add any symbols from globals
            for name, value in self._globals.items():
                if isinstance(value, (int, float)):
                    local_dict[name] = value
                elif callable(value):
                    # Keep callable functions
                    pass

            self._sympy_expr = sp.sympify(self.expression_str, locals=local_dict)

            # Find all free symbols (parameters we need values for)
            self._free_symbols = self._sympy_expr.free_symbols - {self._indep_sym}
            self._param_names = [str(s) for s in self._free_symbols]

        except Exception:
            # Fall back to direct evaluation if sympy parsing fails
            self._sympy_expr = None
            self._free_symbols = set()
            self._param_names = []

    def set_parameters(self, **kwargs):
        """Set parameter values for the function."""
        self._parameters.update(kwargs)

    def vectorize(self, globals_dict=None):
        """Enable vectorized evaluation with numpy arrays."""
        self._vectorized = True
        if globals_dict:
            self._globals.update(globals_dict)

    def __call__(self, x):
        """
        Evaluate the function at x.

        Parameters
        ----------
        x : float or array-like
            The value(s) at which to evaluate the function

        Returns
        -------
        float or ndarray
            The function value(s)
        """
        if self._sympy_expr is not None:
            return self._eval_sympy(x)
        else:
            return self._eval_direct(x)

    def _eval_sympy(self, x):
        """Evaluate using sympy lambdify for speed."""
        # Build substitution dict for parameters
        subs = {}
        for sym in self._free_symbols:
            name = str(sym)
            if name in self._parameters:
                subs[sym] = self._parameters[name]
            elif name in self._globals:
                subs[sym] = self._globals[name]

        # Substitute parameters
        expr = self._sympy_expr.subs(subs)

        # Check if expression still has free symbols (other than independent var)
        remaining = expr.free_symbols - {self._indep_sym}
        if remaining:
            raise ValueError(
                f"Undefined parameters: {[str(s) for s in remaining]}. "
                f"Use set_parameters() or pass globals."
            )

        # Use lambdify for efficient numpy evaluation
        if self._vectorized or isinstance(x, np.ndarray):
            f = sp.lambdify(self._indep_sym, expr, modules=["numpy"])
            return f(x)
        else:
            # For scalar evaluation
            return float(expr.subs(self._indep_sym, x))

    def _eval_direct(self, x):
        """Fallback direct evaluation using eval."""
        # Build namespace for evaluation
        namespace = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "pi": np.pi,
            "e": np.e,
            "abs": np.abs,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "arcsin": np.arcsin,
            "arccos": np.arccos,
            "arctan": np.arctan,
            self.independent_variable: x,
        }
        namespace.update(self._globals)
        namespace.update(self._parameters)

        return eval(self.expression_str, {"__builtins__": {}}, namespace)

    def __repr__(self):
        return f"StringFunction('{self.expression_str}', independent_variable='{self.independent_variable}')"
