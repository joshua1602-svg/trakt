#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_text.py

Channel-safe text normalisation for the Copilot surface
(``mi_agent_api.copilot_text``) — the fix for the black-square / replacement-glyph
artefacts that appeared after values and sentences in Copilot responses.

The inputs below are the exact categories of string the MI Agent emits and that
rendered as squares: em/en dashes and minus signs in answers, the arrow used in
provenance and query-trace notes, the Greek deltas and sigmas used in metric
labels, non-breaking spaces between a value and its unit, zero-width and
private-use codepoints, C0/C1 control characters, and malformed citation
remnants.

What must survive untouched is asserted just as hard: currency signs,
percentages, decimal punctuation, dates, bullets and useful Markdown — and every
numeric value.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from mi_agent_api.copilot_text import normalise_payload, normalise_text

#: Everything Copilot rendered as a black square must be gone afterwards.
_UNRENDERABLE = "—–−→←⇒≤≥≈≠∈⊆×÷±ΔΣ✓✗⚠…·•​‌‍﻿­"


def _has_no_squares(text: str) -> bool:
    return not any(ch in _UNRENDERABLE for ch in text)


# --------------------------------------------------------------------------- #
# The exact categories that rendered as black squares
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    # em / en dash and minus sign in answers and KPI labels
    ("Total balance is £8.9m — across 73 loans", "Total balance is £8.9m - across 73 loans"),
    ("2026-01 – 2026-06", "2026-01 - 2026-06"),
    ("Change: −£1,204,553.20", "Change: -£1,204,553.20"),
    # arrows in provenance / query-trace notes
    ("parser → executor → renderer", "parser -> executor -> renderer"),
    ("KFI ⇒ Application ⇒ Offer", "KFI => Application => Offer"),
    # Greek decorations on metric labels
    ("Δ balance", "Delta balance"),
    ("Σ exposure", "Sum exposure"),
    # relational operators in warnings and limit tests
    ("Coverage ≥ 95%", "Coverage >= 95%"),
    ("LTV ≤ 60%", "LTV <= 60%"),
    ("approx ≈ 12.5%", "approx ~ 12.5%"),
    # separators used between facts
    ("WA LTV 62.4% · 73 loans", "WA LTV 62.4% - 73 loans"),
    ("Truncated…", "Truncated..."),
    # arithmetic decorations
    ("3 × 4 ÷ 2 ± 1", "3 x 4 / 2 +/- 1"),
    # smart quotes
    ("the “funded” book", 'the "funded" book'),
])
def test_unrenderable_characters_become_plain_equivalents(raw, expected):
    out = normalise_text(raw)
    assert out == expected
    assert _has_no_squares(out)


def test_zero_width_characters_are_removed():
    raw = "£8​,9‌00‍,000﻿"
    assert normalise_text(raw) == "£8,900,000"


def test_soft_hyphen_and_word_joiner_are_removed():
    assert normalise_text("con­centra⁠tion") == "concentration"


def test_non_breaking_spaces_become_ordinary_spaces():
    assert normalise_text("£100 000") == "£100 000"
    assert normalise_text("62.4 %") == "62.4 %"
    assert " " not in normalise_text("73 loans")


def test_private_use_characters_are_removed():
    # U+F0B7 is the Symbol-font bullet Word/Copilot renders as a square.
    assert normalise_text(" Nottingham") == "Nottingham"
    assert normalise_text("hidden") == "hidden"
    assert normalise_text("a\U000f0000b") == "ab"


def test_control_characters_are_removed_but_whitespace_is_kept():
    assert normalise_text("bal\x07ance") == "balance"
    assert normalise_text("a\x00b\x1fc\x9fd") == "abcd"
    assert normalise_text("line one\nline two") == "line one\nline two"
    assert normalise_text("col\tvalue") == "col\tvalue"


def test_variation_selectors_are_removed():
    assert normalise_text("warning ⚠️ now") == "warning [!] now"


@pytest.mark.parametrize("raw", [
    "【4:0†source】The balance is £8.9m",
    "The balance is £8.9m【ref】",
    "The balance is £8.9m [[cite: platform_canonical]]",
    "The balance is £8.9m [cite:run_2026_01]",
])
def test_malformed_citation_remnants_are_removed(raw):
    out = normalise_text(raw)
    assert out == "The balance is £8.9m"
    assert "【" not in out and "cite" not in out


# --------------------------------------------------------------------------- #
# What must be preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "£8,900,000.00",
    "62.4%",
    "£1.2m at 4.75%",
    "2026-01-31",
    "31 January 2026",
    "- first bullet\n- second bullet",
    "* starred bullet",
    "**bold** and _italic_ and `code`",
    "| col | value |\n| --- | ----- |",
    "Café Nero",           # combining accent (NFKC composes, never drops)
    "€1,000 / $2,000 / ¥3,000",
])
def test_useful_content_is_preserved(text):
    out = normalise_text(text)
    # Currency, percent, digits and separators all survive.
    for ch in "£%€$¥0123456789.,":
        assert text.count(ch) == out.count(ch), f"{ch!r} changed in {text!r} -> {out!r}"


def test_currency_and_percent_survive_a_full_answer():
    raw = ("Largest geographic concentration: Nottingham at £831,402 "
           "(15.4% of the book) across 10 ITL3 area(s). Basis: collateral; "
           "resolved coverage 100.0%.")
    out = normalise_text(raw)
    assert "£831,402" in out and "15.4%" in out and "100.0%" in out


def test_normalisation_is_idempotent():
    raw = "Δ balance −£1.2m — parser → executor​ (≥95%)"
    once = normalise_text(raw)
    assert normalise_text(once) == once


def test_numeric_values_are_never_mutated():
    payload = {"balance": 8_900_000.25, "count": 73, "pct": 15.4,
               "flag": True, "missing": None,
               "label": "Nottingham — ITL3"}
    out = normalise_payload(payload)
    assert out["balance"] == 8_900_000.25
    assert out["count"] == 73 and isinstance(out["count"], int)
    assert out["pct"] == 15.4
    assert out["flag"] is True
    assert out["missing"] is None
    assert out["label"] == "Nottingham - ITL3"


def test_payload_normalisation_is_recursive_and_keeps_keys():
    payload = {"rows": [{"area​": "Bristol — BS", "balance": 1234.5}],
               "notes": ["parser → executor"]}
    out = normalise_payload(payload)
    # Dict KEYS are contract identifiers and are left untouched.
    assert list(out["rows"][0].keys()) == ["area​", "balance"]
    assert out["rows"][0]["area​"] == "Bristol - BS"
    assert out["rows"][0]["balance"] == 1234.5
    assert out["notes"] == ["parser -> executor"]


def test_empty_and_non_string_inputs_are_safe():
    assert normalise_text("") == ""
    assert normalise_payload(None) is None
    assert normalise_payload(42) == 42
    assert normalise_payload([]) == []


# --------------------------------------------------------------------------- #
# End-to-end: no artefact survives the Copilot route
# --------------------------------------------------------------------------- #
def test_copilot_response_carries_no_unrenderable_characters(monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from mi_agent_api import data_source
    from mi_agent_api.app import app

    demo = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))[0]
    monkeypatch.setenv("MI_AGENT_DATA_CSV", str(demo))
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    monkeypatch.setenv("TRAKT_COPILOT_AUTH_MODE", "disabled")
    data_source.reset_cache()
    try:
        for question in ("What is the total current balance?",
                         "Where is the book most concentrated by region?",
                         "Show balance by LTV bucket.",
                         "Show the top 10 loans by current balance."):
            r = TestClient(app).post("/v1/copilot/mi/query",
                                     json={"question": question})
            assert r.status_code == 200, r.text
            body = json.dumps(r.json(), ensure_ascii=False)
            offenders = sorted({ch for ch in body if ch in _UNRENDERABLE})
            assert not offenders, f"{question!r} still renders: {offenders}"
    finally:
        data_source.reset_cache()
