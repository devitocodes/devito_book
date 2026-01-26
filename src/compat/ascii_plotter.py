"""
Plotter: A replacement for scitools.avplotter.Plotter.

Provides simple ASCII plotting in the terminal for quick visualization
without requiring a graphical display.
"""


class Plotter:
    """
    ASCII plotter for terminal-based visualization.

    Creates simple ASCII plots that can be printed line-by-line,
    useful for monitoring long-running simulations in a terminal.

    Parameters
    ----------
    ymin : float
        Minimum y-axis value
    ymax : float
        Maximum y-axis value
    width : int, optional
        Width of the plot in characters (default: 60)
    symbols : str, optional
        Characters to use for plotting points (default: '+o')
        First character for first value, second for second value, etc.

    Examples
    --------
    >>> p = Plotter(ymin=-1, ymax=1, width=40, symbols='+o')
    >>> print(p.plot(0.0, 0.5, 0.48))
    '                    +o                  '
    """

    def __init__(self, ymin, ymax, width=60, symbols="+o"):
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.symbols = symbols

    def _y_to_pos(self, y):
        """Convert a y value to a position in the plot width."""
        if self.ymax == self.ymin:
            return self.width // 2

        # Clamp y to range
        y = max(self.ymin, min(self.ymax, y))

        # Map to position
        fraction = (y - self.ymin) / (self.ymax - self.ymin)
        pos = int(fraction * (self.width - 1))
        return pos

    def plot(self, t, *values):
        """
        Create an ASCII plot line for the given values.

        Parameters
        ----------
        t : float
            The current time (not used in plot, but part of interface)
        *values : float
            Values to plot. Each value gets a different symbol.

        Returns
        -------
        str
            ASCII string representation of the plot line
        """
        # Start with empty line
        line = [" "] * self.width

        # Plot each value with its symbol
        for i, y in enumerate(values):
            if y is None:
                continue
            pos = self._y_to_pos(y)
            symbol = self.symbols[i % len(self.symbols)]
            line[pos] = symbol

        return "".join(line)

    def __repr__(self):
        return (
            f"Plotter(ymin={self.ymin}, ymax={self.ymax}, "
            f"width={self.width}, symbols='{self.symbols}')"
        )
