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
        # "gbp" is a legacy UNIT TAG in the approved test library, not a claim
        # about the reporting currency. The symbol comes from the governed
        # currency in force, so a EUR book shows EUR headroom.
        from mi_agent_api.insight_generators import money as _governed
        return _governed(v)
    if u in ("count", "loans", "number"):
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def format_headroom(value: Optional[float], unit: Optional[str]) -> str:
    """Distance to the limit, in the unit that distance is actually measured in.

    Headroom is a DIFFERENCE between two values, not a value. For a percentage
    test that difference is percentage points, and printing it with a percent
    sign puts two incompatible percentages on the same line: a reader who has
    just been told London is 47% utilised then reads "16.0% of headroom" as a
    share of something. It is 16.0 points of exposure share. Currency and count
    tests have no such ambiguity — a difference of pounds is pounds — so they
    keep their own formatting unchanged.
    """
    v = _num(value)
    if v is None:
        return "—"
    u = str(unit or "").strip().lower()
    if u in ("percent", "percentage", "pct", "%", "percentage_points", "pp"):
        return f"{v:.1f}pp"
    return format_measure(v, unit)


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
            # The governed limit DIRECTION. "max" is a ceiling, "min" a floor;
            # the evaluator already computes status and utilisation for both, so
            # nothing here may assume that a higher value is the worse one.
            "operator": str(t.get("operator") or "max").lower(),
            # The operator-configured display priority, where the configuration
            # expresses one. Ranking falls back to it before the id tie-break.
            "severity": t.get("severity"),
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


def attach_history(rows: Sequence[Dict[str, Any]],
                   history: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add the PRIOR governed value of each test, where one was evaluated.

    A covenant table states where a test sits and where it is expected to go,
    and leaves the reader to guess whether it has been moving toward the limit
    or away from it. The prior point comes from ``compute_history``, which
    evaluates TODAY's approved configuration against each historical frame — so
    prior and current are comparable, and a change is a change in the book
    rather than a change in the definition.

    Rows are copied; nothing is recomputed. A test with fewer than two governed
    frames simply carries no prior, and the presentation layer shows nothing
    rather than inventing a direction from one point.
    """
    out = [dict(r) for r in rows]
    if not (history or {}).get("available"):
        return out
    by_test: Dict[Any, Mapping[str, Any]] = {
        sr.get("testId"): sr for sr in (history or {}).get("series") or ()
        if isinstance(sr, Mapping)}
    for row in out:
        sr = by_test.get(row.get("test_id"))
        points = [p for p in ((sr or {}).get("points") or ())
                  if isinstance(p, Mapping) and _num(p.get("value")) is not None]
        if len(points) < 2:
            continue
        prior = points[-2]
        row["prior_value"] = _num(prior.get("value"))
        row["prior_date"] = prior.get("reportingDate")
        row["prior_status"] = normalise_status(prior.get("status"))
        row["periods_observed"] = len(points)
        # The engine's classification travels with the row.
        row["direction"] = sr.get("direction")
    return out


def attach_stress(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add the governed stress effect to each row."""
    from mi_agent_api.concentration_tests_api import stress_effect

    out = []
    for row in rows:
        row = dict(row)
        row["stress_effect"] = stress_effect(
            row.get("value"), row.get("stress_value"), row.get("limit"),
            row.get("operator", "max"))
        out.append(row)
    return out


#: The engine's direction codes, in the words the page uses. The engine decides
#: WHICH WAY a test moved — against its limit, so a floor test falling is
#: deteriorating; this layer decides only what to call that.
_TRAVEL_WORDING = {
    "toward_limit": "toward the limit",
    "away_from_limit": "away from the limit",
    "broadly_unchanged": "broadly unchanged",
}

_STRESS_WORDING = {
    "eases": ("converting the whole pipeline would dilute this test, "
              "not stress it"),
    "no_effect": "the stress does not move this test",
}


def travel(row: Mapping[str, Any]) -> Optional[str]:
    """The governed direction of travel, in words, or ``None``.

    Reads ``direction`` from the concentration history service. It is NOT
    recomputed here: which way is worse is a property of the governed operator,
    and a presentation layer that decided it from the number alone inverted
    every minimum-type test.
    """
    return _TRAVEL_WORDING.get(str(row.get("direction") or ""))


def stress_note(row: Mapping[str, Any]) -> Optional[str]:
    """What the stress did, in words, where it did not behave like one."""
    return _STRESS_WORDING.get(str(row.get("stress_effect") or ""))


def rank_key(row: Mapping[str, Any]):
    """Deterministic severity order, exactly as specified.

    current breach → current warning → expected forecast breach → shortest
    breach horizon → stress breach → highest utilisation → configured display
    priority → stable test id.

    Every level is derived from the governed payload; none of it is client
    specific, and the final tie-break on ``test_id`` means two runs over the
    same configuration can never disagree on order.
    """
    util = row.get("utilisation")
    horizon = row.get("breach_horizon")
    return (
        0 if row.get("status") == STATUS_BREACH else 1,
        0 if row.get("status") == STATUS_WARNING else 1,
        0 if row.get("expected_breach") else 1,
        # An earlier horizon outranks a later one; absent sorts last.
        str(horizon) if horizon else "~",
        0 if row.get("stress_breach") else 1,
        -(util if util is not None else -1.0),
        _priority_rank(row.get("severity")),
        str(row.get("test_id") or row.get("label") or ""),
    )


#: Configured display priority, most severe first. An unrecognised or absent
#: value sorts after everything the configuration ranked explicitly.
_PRIORITY_ORDER = ("critical", "high", "material", "medium", "low", "informational")


def _priority_rank(severity: Any) -> int:
    value = str(severity or "").strip().lower()
    return _PRIORITY_ORDER.index(value) if value in _PRIORITY_ORDER else len(_PRIORITY_ORDER)


def select_tests(rows: Sequence[Mapping[str, Any]],
                 limit: int = MAX_TESTS_ON_SLIDE) -> List[Dict[str, Any]]:
    """The most important tests, ranked. Showing every test would produce an
    unreadable slide; showing an arbitrary subset would be worse."""
    ordered = sorted((dict(r) for r in rows), key=rank_key)
    return ordered[:limit]


def overflow(rows: Sequence[Mapping[str, Any]],
             limit: int = MAX_TESTS_ON_SLIDE) -> List[Dict[str, Any]]:
    """Ranked tests beyond the primary slide's capacity.

    Squeezing forty tests onto one page produces an unreadable slide; dropping
    them silently produces a misleading one. They are returned so the deck can
    carry them on a second page or disclose them in metadata.
    """
    return sorted((dict(r) for r in rows), key=rank_key)[limit:]


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
        # Tests that breach ONLY under stress. Counted per test rather than by
        # subtracting one total from the other: whether the stress set contains
        # the forecast set depends on the configuration, and on a floor-limit or
        # a run-off test it need not.
        "stress_only_breaches": sum(1 for r in rows if r.get("stress_breach")
                                    and not r.get("expected_breach")),
        "tightest": min((r for r in rows if r.get("headroom") is not None),
                        key=lambda r: r["headroom"], default=None),
        # The test CLOSEST TO BREACHING, which is not the same as the one with
        # the least headroom: headroom is in the test's own unit, so £2m of room
        # on a £2bn limit and 5pp on a 95pp limit cannot be ranked against each
        # other. Utilisation is the evaluator's own unitless "how close am I",
        # and it already accounts for the limit's direction — a floor test at
        # 8 against a minimum of 10 is over-utilised, not under.
        "closest": max((r for r in rows if r.get("utilisation") is not None),
                       key=lambda r: r["utilisation"], default=None),
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
