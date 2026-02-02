import re

import numpy as np
import pytest


def _devito_importable() -> bool:
    try:
        import devito  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.devito
@pytest.mark.skipif(not _devito_importable(), reason="Devito not importable in this environment")
def test_readme_devito_example_executes():
    """Ensure README's Devito example is runnable and uses an assignable update."""
    readme = open("README.md", encoding="utf-8").read()
    match = re.search(r"## What is Devito\?\s+.*?```python\s*\n(.*?)```", readme, re.S)
    assert match, "Could not locate the 'What is Devito?' python code block in README.md"

    code = match.group(1)
    # Safety/intent checks: we want to ensure the README teaches the right Devito pattern.
    assert "solve(" in code
    assert "u.forward" in code

    namespace: dict[str, object] = {}
    exec(compile(code, "README.md::what-is-devito", "exec"), namespace)


def test_first_pde_explanation_matches_tested_snippet():
    """Ensure narrative doesn't claim u.data[1]=u.data[0] for 2nd-order wave scheme."""
    text = open("chapters/devito_intro/first_pde.qmd", encoding="utf-8").read()
    assert "same as t=0" not in text
    assert "second-order accuracy" in text or "2nd-order accuracy" in text
    assert "0.5 * dt**2" in text


def test_elliptic_l1norm_is_relative_change():
    """Ensure elliptic chapter uses a standard relative-change criterion."""
    text = open("chapters/elliptic/elliptic.qmd", encoding="utf-8").read()
    assert "p_{i,j}^{(k+1)} - p_{i,j}^{(k)}" in text
    assert "np.abs(p.data[:] - pn.data[:])" in text

    # Guard against the previous cancellation-prone definition.
    assert "np.abs(p.data[:]) - np.abs(pn.data[:])" not in text

    p_prev = np.array([1.0, 1.0])
    p_curr = np.array([-1.0, 1.0])
    old = np.sum(np.abs(p_curr) - np.abs(p_prev)) / np.sum(np.abs(p_prev))
    new = np.sum(np.abs(p_curr - p_prev)) / (np.sum(np.abs(p_prev)) + 1.0e-16)
    assert old == pytest.approx(0.0)
    assert new > 0.0
