"""
Compatibility module providing drop-in replacements for deprecated scitools components.

This module provides modern implementations of:
- StringFunction: Parse and evaluate mathematical string expressions
- MovingPlotWindow: Windowed plotting for long time series
- Plotter: ASCII terminal plotting
- movie: Create HTML animations from image sequences
"""

from .ascii_plotter import Plotter
from .movie import movie
from .moving_plot_window import MovingPlotWindow
from .string_function import StringFunction

__all__ = ["MovingPlotWindow", "Plotter", "StringFunction", "movie"]
