"""
MovingPlotWindow: A replacement for scitools.MovingPlotWindow.MovingPlotWindow.

Provides windowed plotting for long time series, showing only a portion
of the data at a time for clearer visualization.
"""


class MovingPlotWindow:
    """
    Manage a moving plot window for visualizing long time series.

    The window shows a fixed time interval, sliding forward as the
    simulation progresses. This is useful for very long simulations
    where showing all data at once would make the plot unreadable.

    Parameters
    ----------
    window_width : float
        The width of the plot window in time units
    dt : float
        The time step between data points
    yaxis : list of float, optional
        [ymin, ymax] for fixed y-axis limits
    mode : str, optional
        'continuous drawing' for smooth sliding window (default)

    Examples
    --------
    >>> pm = MovingPlotWindow(window_width=2.0, dt=0.01, yaxis=[-1, 1])
    >>> for n in range(1000):
    ...     if pm.plot(n):
    ...         # Make plot from pm.first_index_in_plot to n
    ...         pass
    ...     pm.update(n)
    """

    def __init__(self, window_width, dt, yaxis=None, mode="continuous drawing"):
        self.window_width = window_width
        self.dt = dt
        self.yaxis = yaxis if yaxis is not None else [None, None]
        self.mode = mode

        # Number of points in the window
        self.window_points = int(window_width / dt)

        # Current state
        self._current_n = 0
        self._first_index = 0

    @property
    def first_index_in_plot(self):
        """Return the first index that should be included in the current plot."""
        return self._first_index

    def plot(self, n):
        """
        Determine if a plot should be made at step n.

        Parameters
        ----------
        n : int
            Current time step index

        Returns
        -------
        bool
            True if a plot should be made at this step
        """
        # Always plot (the caller can decide to skip frames for performance)
        return True

    def axis(self):
        """
        Return the axis limits for the current window.

        Returns
        -------
        list
            [xmin, xmax, ymin, ymax] for use with plt.axis()
        """
        t_start = self._first_index * self.dt
        t_end = t_start + self.window_width
        return [t_start, t_end, self.yaxis[0], self.yaxis[1]]

    def update(self, n):
        """
        Update the window state after plotting step n.

        Parameters
        ----------
        n : int
            Current time step index (just plotted)
        """
        self._current_n = n

        # Slide window if we've exceeded the window width
        if n > self.window_points:
            self._first_index = n - self.window_points
        else:
            self._first_index = 0

    def __repr__(self):
        return (
            f"MovingPlotWindow(window_width={self.window_width}, "
            f"dt={self.dt}, yaxis={self.yaxis})"
        )
