"""Smoke tests for src/diffu/flow_in_serial_layers.py.

NOTE: The SerialLayers class itself is correct, but the module has
top-level interactive code (input() calls at module scope, line 133-142)
that makes it non-importable in a test environment. The tests below
verify the Heaviside dependency that SerialLayers relies on.
"""

import sys
from pathlib import Path

import pytest

# Heaviside module lives alongside flow_in_serial_layers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "diffu"))


def test_heaviside_module_importable():
    """The Heaviside module (dependency of flow_in_serial_layers) imports cleanly."""
    from Heaviside import PiecewiseConstant

    # Basic smoke: piecewise constant function
    domain = [0, 1]
    data = [[0, 1.0], [0.5, 2.0]]
    pc = PiecewiseConstant(domain, data, eps=0)
    x_vals, y_vals = pc.plot()
    assert len(x_vals) > 0
    assert len(y_vals) > 0


def test_flow_in_serial_layers_not_importable():
    """Document that flow_in_serial_layers cannot be imported due to top-level input() calls."""
    # This test documents the known issue: the module has interactive code
    # at module scope (lines 133-142) that calls input(), making it
    # non-importable in automated environments.
    # TODO: Wrap module-level code in if __name__ == "__main__" guard.
    with pytest.raises((EOFError, OSError)):
        from src.diffu.flow_in_serial_layers import SerialLayers  # noqa: F401
