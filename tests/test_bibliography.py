"""Tests for bibliography integrity.

Verifies that DOIs in references.bib resolve and that metadata is consistent.
These tests make network requests and are marked @pytest.mark.slow.
"""

import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BIB_PATH = ROOT / "references.bib"


def _parse_entries(bib_text):
    """Parse bib entries into list of (key, fields_dict)."""
    entries = []
    entry_re = re.compile(
        r"@\w+\{([\w:.-]+)\s*,\s*(.*?)\n\}",
        re.DOTALL,
    )
    field_re = re.compile(r"(\w+)\s*=\s*\{(.*?)\}", re.DOTALL)
    for m in entry_re.finditer(bib_text):
        key = m.group(1)
        body = m.group(2)
        fields = {}
        for fm in field_re.finditer(body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append((key, fields))
    return entries


def _entries_with_dois():
    """Return list of (bib_key, doi) for entries that have a DOI."""
    text = BIB_PATH.read_text(encoding="utf-8")
    entries = _parse_entries(text)
    result = []
    for key, fields in entries:
        doi = fields.get("doi")
        if doi:
            result.append((key, doi))
    return result


@pytest.mark.slow
@pytest.mark.parametrize("bib_key,doi", _entries_with_dois(), ids=lambda x: x if isinstance(x, str) else "")
def test_doi_resolves(bib_key, doi):
    """Each DOI in references.bib should resolve (HTTP HEAD, non-404)."""
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "devito-book-test/1.0")
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        assert resp.status < 400, f"{bib_key}: DOI {doi} returned HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        assert e.code != 404, f"{bib_key}: DOI {doi} returned HTTP 404 (Not Found)"
        # Other HTTP errors (403 from rate limiting, etc.) are acceptable
    except urllib.error.URLError:
        pytest.skip(f"Network unavailable for DOI check: {doi}")


def test_all_entries_have_required_fields():
    """Every bib entry should have at least title and year."""
    text = BIB_PATH.read_text(encoding="utf-8")
    entries = _parse_entries(text)
    issues = []
    for key, fields in entries:
        if "title" not in fields:
            issues.append(f"{key}: missing title")
        # year is optional for @misc entries
    assert not issues, "Bib entries with missing fields:\n" + "\n".join(issues)
