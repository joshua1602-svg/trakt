"""mi_agent_pptx.concentration — presenting the governed concentration tests.

Adapts the ``/mi/concentration-tests`` envelope
(:func:`mi_agent_api.concentration_tests_api.compute_concentration_tests`) into
the handful of rows an investor slide can carry.

**No concentration methodology lives here.** Values, thresholds, statuses,
utilisation, headroom, breach flags and the forward states are all produced by
the governed evaluator. This module only *selects* which tests are worth a slide
and *formats* them. If it ever needs a number the evaluator does not supply, the
correct fix is in the evaluator, not here.

Three states are kept rigorously separate, because collapsing them is the
failure this section exists to prevent:

  * **Current** — funded actuals at the funded reporting date.
  * **Expected** — the governed forecast model's projection.
  * **Stress**   — the all-pipeline-converts maximum-exposure case. It is a
    stress, never an expectation, and is labelled as such everywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

#: The approved vocabulary, plus the legacy monitor's colours mapped onto it.
STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_BREACH = "breach"
STATUS_UNAVAILABLE = "unavailable"

_STATUS_NORMAL = {
    "pass": STATUS_PASS, "green": STATUS_PASS, "ok": STATUS_PASS,
    "warning": STATUS_WARNING, "amber": STATUS_WARNING, "warn": STATUS_WARNING,
    "breach": STATUS_BREACH, "red": STATUS_BREACH, "fail": STATUS_BREACH,
    "needs_review": STATUS_UNAVAILABLE, "unavailable": STATUS_UNAVAILABLE,
    "indicative_only": STATUS_UNAVAILABLE,
}

#: How many tests a single slide can carry and stay readable.
MAX_TESTS_ON_SLIDE = 5

#: Sources the envelope may report. ``legacy_extracted`` is explicitly NOT
#: operator-approved and must be disclosed as such wherever it is shown.
SOURCE_APPROVED = "approved_configuration"
SOURCE_LEGACY = "legacy_extracted"


def normalise_status(value: Any) -> str:
    return _STATUS_NORMAL.get(str(value or "").strip().lower(), STATUS_UNAVAILABLE)


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_measure(value: Optional[float], unit: Optional[str]) -> str:
    """A governed test value in its own unit — never re-scaled."""
    v = _num(value)
    if v is None:
        return "—"
    u = str(unit or "").strip().lower()
    if u in ("percent", "percentage", "pct", "%", "percentage_points", "pp"):
        return f"{v:.1f}%"
    if u in ("gbp", "currency", "amount"):
        a = abs(v)
        if a >= 1e9:
            return f"£{v / 1e9:.2f}bn"
        if a >= 1e6:
            return f"£{v / 1e6:.1f}m"
        if a >= 1e3:
            return f"£{v / 1e3:.0f}k"
        return f"£{v:,.0f}"
    if u in ("count", "loans", "number"):
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def _state(row: Mapping[str, Any], key: str) -> Dict[str, Any]:
    block = row.get(key)
    return dict(block) if isinstance(block, Mapping) else {}


def adapt_tests(envelope: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every governed test as a flat presentation row (unranked, unfiltered)."""
    rows: List[Dict[str, Any]] = []
    for t in (envelope or {}).get("tests") or ():
        if not isinstance(t, Mapping):
            continue
        expected = _state(t, "expected")
        stress = _state(t, "fullPipeline")
        horizon = _state(t, "expectedBreachHorizon")
        rows.append({
            "test_id": t.get("testId"),
            "label": str(t.get("displayName") or t.get("metricId") or ""),
            "category": t.get("category") or "",
            "unit": t.get("unit"),
            # -- current (the only actual) -------------------------------
            "value": _num(t.get("currentValue")),
            "limit": _num(t.get("threshold")),
            "utilisation": _num(t.get("utilization")),
            "headroom": _num(t.get("headroom")),
            "status": normalise_status(t.get("status")),
            "reporting_date": t.get("reportingDate"),
            "data_status": t.get("dataStatus"),
            # -- expected forecast ----------------------------------------
            "expected_value": _num(expected.get("value")),
            "expected_utilisation": _num(expected.get("utilization")),
            "expected_headroom": _num(expected.get("headroom")),
            "expected_status": normalise_status(expected.get("status")),
            "expected_breach": bool(t.get("expectedBreach")),
            "breach_horizon": horizon.get("period"),
            # -- all-pipeline-converts stress ------------------------------
            "stress_value": _num(stress.get("value")),
            "stress_utilisation": _num(stress.get("utilization")),
            "stress_headroom": _num(stress.get("headroom")),
            "stress_status": normalise_status(stress.get("status")),
            "stress_breach": bool(t.get("fullPipelineBreach")),
            "forecast_treatment": t.get("forecastTreatment"),
        })
    return rows


def rank_key(row: Mapping[str, Any]):
    """Deterministic severity order, exactly as specified:

    current breach → forecast breach → amber proximity → expected breach
    horizon → utilisation. Ties break on the label so two runs never disagree.
    """
    current_breach = row.get("status") == STATUS_BREACH
    forecast_breach = bool(row.get("expected_breach"))
    amber = row.get("status") == STATUS_WARNING or \
        row.get("expected_status") == STATUS_WARNING
    # An earlier horizon outranks a later one; absent horizon sorts last.
    horizon = row.get("breach_horizon")
    horizon_key = str(horizon) if horizon else "~"
    util = row.get("utilisation")
    return (
        0 if current_breach else 1,
        0 if forecast_breach else 1,
        0 if amber else 1,
        horizon_key,
        -(util if util is not None else -1.0),
        str(row.get("label", "")),
    )


def select_tests(rows: Sequence[Mapping[str, Any]],
                 limit: int = MAX_TESTS_ON_SLIDE) -> List[Dict[str, Any]]:
    """The most important tests, ranked. Showing every test would produce an
    unreadable slide; showing an arbitrary subset would be worse."""
    ordered = sorted((dict(r) for r in rows), key=rank_key)
    return ordered[:limit]


def summarise(envelope: Optional[Mapping[str, Any]],
              rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts for the slide's summary strip, from the governed summary where the
    evaluator provides one and from the rows otherwise."""
    summary = (envelope or {}).get("summary") or {}
    breaches = summary.get("breaches")
    if breaches is None:
        breaches = sum(1 for r in rows if r.get("status") == STATUS_BREACH)
    warnings = summary.get("warnings")
    if warnings is None:
        warnings = sum(1 for r in rows if r.get("status") == STATUS_WARNING)
    return {
        "tests": len(rows),
        "breaches": int(breaches or 0),
        "warnings": int(warnings or 0),
        "expected_breaches": int(summary.get("expectedBreaches")
                                 or sum(1 for r in rows if r.get("expected_breach"))),
        "stress_breaches": int(summary.get("fullPipelineBreaches")
                               or sum(1 for r in rows if r.get("stress_breach"))),
        "tightest": min((r for r in rows if r.get("headroom") is not None),
                        key=lambda r: r["headroom"], default=None),
    }


def is_available(envelope: Optional[Mapping[str, Any]]) -> bool:
    """True when governed concentration tests exist for this scope."""
    return bool((envelope or {}).get("tests"))


def forward_states_available(envelope: Optional[Mapping[str, Any]]) -> bool:
    """True when the Expected / Full-Pipeline states were actually evaluated.

    Without an active in-scope pipeline the evaluator declines to project rather
    than restating funded as a forecast — so the deck must not draw those marks.
    """
    return bool(((envelope or {}).get("states") or {}).get("available"))


def source_disclosure(envelope: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The one line the slide must carry about where these limits came from."""
    source = (envelope or {}).get("source")
    if source == SOURCE_APPROVED:
        lineage = (envelope or {}).get("lineage") or {}
        version = lineage.get("configurationVersion")
        return ("Operator-approved concentration configuration"
                + (f" v{version}" if version else ""))
    if source == SOURCE_LEGACY:
        return ("Extracted limit monitor — not an operator-approved "
                "concentration configuration")
    return None
