import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


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

    # Correct pattern: np.abs(VAR.data[:] - VAR.data[:])
    correct_pattern = re.compile(r"np\.abs\(\w+\.data\[:\]\s*-\s*\w+\.data\[:\]\)")
    assert correct_pattern.search(text), "Chapter must use np.abs(a.data[:] - b.data[:])"

    # Guard against the previous cancellation-prone definition:
    # np.abs(VAR.data[:]) - np.abs(VAR.data[:])
    wrong_pattern = re.compile(r"np\.abs\(\w+\.data\[:\]\)\s*-\s*np\.abs\(\w+\.data\[:\]\)")
    assert not wrong_pattern.search(text), "Chapter must NOT use np.abs(a.data[:]) - np.abs(b.data[:])"

    # Also check source file
    src_text = open("src/elliptic/laplace_devito.py", encoding="utf-8").read()
    assert correct_pattern.search(src_text), "Source must use np.abs(a.data[:] - b.data[:])"
    assert not wrong_pattern.search(src_text), "Source must NOT use np.abs(a.data[:]) - np.abs(b.data[:])"

    # Numerical proof that old formula is wrong
    p_prev = np.array([1.0, 1.0])
    p_curr = np.array([-1.0, 1.0])
    old = np.sum(np.abs(p_curr) - np.abs(p_prev)) / np.sum(np.abs(p_prev))
    new = np.sum(np.abs(p_curr - p_prev)) / (np.sum(np.abs(p_prev)) + 1.0e-16)
    assert old == pytest.approx(0.0)
    assert new > 0.0


# ============================================================================
# Include directive and citation consistency tests
# ============================================================================

def _collect_qmd_files():
    """Collect all .qmd files under chapters/."""
    return sorted(ROOT.glob("chapters/**/*.qmd"))


def _parse_bib_keys(bib_path):
    """Extract all citation keys from a .bib file."""
    text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{(\w[\w:.-]*)", text))


def test_include_directives_resolve():
    """Every {{< include ... >}} directive in chapter .qmd files must resolve.

    Only checks top-level chapter files (not snippet .qmd files that are
    themselves included, since Quarto resolves nested includes differently).
    """
    include_re = re.compile(r"\{\{<\s*include\s+(.*?)\s*>\}\}")
    missing = []
    for qmd in _collect_qmd_files():
        # Skip snippet files — they are nested includes resolved by Quarto
        # from the parent chapter's directory, not their own.
        if "/snippets/" in str(qmd):
            continue
        text = qmd.read_text(encoding="utf-8")
        for m in include_re.finditer(text):
            target = m.group(1).strip().strip('"').strip("'")
            # Try resolving relative to the file's directory
            resolved = (qmd.parent / target).resolve()
            # Also try resolving relative to project root (Quarto behavior)
            resolved_root = (ROOT / target).resolve()
            if not resolved.exists() and not resolved_root.exists():
                missing.append(f"{qmd.relative_to(ROOT)}:{target}")
    assert not missing, "Broken include directives:\n" + "\n".join(missing)


def test_citation_keys_exist_in_bib():
    """Every [@key] used in chapters must exist in references.bib."""
    bib_keys = _parse_bib_keys(ROOT / "references.bib")
    cite_re = re.compile(r"\[@([\w:.-]+)")
    missing = []
    for qmd in _collect_qmd_files():
        text = qmd.read_text(encoding="utf-8")
        for m in cite_re.finditer(text):
            key = m.group(1)
            # Skip cross-reference prefixes (sec-, eq-, fig-, tbl-)
            if key.startswith(("sec-", "eq-", "fig-", "tbl-")):
                continue
            if key not in bib_keys:
                missing.append(f"{qmd.relative_to(ROOT)}: @{key}")
    assert not missing, "Citation keys not in references.bib:\n" + "\n".join(missing)


def test_devito_primary_papers_cited():
    """The Devito primary papers must appear in at least one chapter."""
    cite_re = re.compile(r"\[@[\w:.-]*devito-api[\w:.-]*")
    found = False
    for qmd in _collect_qmd_files():
        text = qmd.read_text(encoding="utf-8")
        if cite_re.search(text):
            found = True
            break
    assert found, "devito-api is never cited in any chapter"
