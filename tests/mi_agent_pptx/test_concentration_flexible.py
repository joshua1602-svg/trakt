"""The concentration slide must work for ANY operator-approved configuration.

Nothing about the presentation may depend on a particular client's tests: not
their names, count, regions, limits, direction, units, or which scenarios their
deployment evaluates. These tests drive the adapter and the ranking directly
with synthetic governed payloads, so a configuration shape can be covered
without standing up a full run for each one.
"""

from __future__ import annotations

import pytest

from mi_agent_pptx import concentration as C


def _test(test_id, name, *, value, limit, status, util, operator="max",
          unit="percent", expected=None, expected_status=None,
          expected_breach=False, horizon=None, stress=None, stress_breach=False,
          severity=None, headroom=None):
    """One governed test payload, in the evaluator's own shape."""
    row = {
        "testId": test_id, "displayName": name, "metricId": "m", "category": "geo",
        "operator": operator, "unit": unit, "severity": severity,
        "currentValue": value, "threshold": limit, "utilization": util,
        "headroom": headroom if headroom is not None else (limit - value),
        "status": status, "reportingDate": "2026-06-30",
        "expectedBreach": expected_breach, "fullPipelineBreach": stress_breach,
    }
    if expected is not None:
        row["expected"] = {"value": expected, "status": expected_status or "pass",
                           "utilization": expected / limit * 100 if limit else None}
    if stress is not None:
        row["fullPipeline"] = {"value": stress, "status": "pass",
                               "utilization": stress / limit * 100 if limit else None}
    if horizon:
        row["expectedBreachHorizon"] = {"period": horizon}
    return row


def _env(tests, *, forward=True, source=C.SOURCE_APPROVED):
    return {"tests": tests, "source": source,
            "states": {"available": forward},
            "lineage": {"configurationVersion": 3},
            "summary": {}}


# --------------------------------------------------------------------------- #
# Configuration shapes.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("count", [1, 3, 5, 9])
def test_any_number_of_tests_is_handled(count):
    rows = C.adapt_tests(_env([
        _test(f"t{i}", f"Test {i}", value=10.0 + i, limit=50.0, status="pass",
              util=(10.0 + i) / 50 * 100) for i in range(count)]))
    assert len(rows) == count
    shown = C.select_tests(rows)
    assert len(shown) == min(count, C.MAX_TESTS_ON_SLIDE)
    # Nothing is lost: whatever does not fit is returned for disclosure.
    assert len(shown) + len(C.overflow(rows)) == count


def test_ranking_follows_the_specified_order():
    """current breach → warning → forecast breach → horizon → stress → util."""
    rows = C.adapt_tests(_env([
        _test("low", "Low utilisation", value=5.0, limit=50.0, status="pass", util=10.0),
        _test("high", "High utilisation", value=45.0, limit=50.0, status="pass", util=90.0),
        _test("stress", "Stress only", value=20.0, limit=50.0, status="pass",
              util=40.0, stress=60.0, stress_breach=True),
        _test("late", "Forecast late", value=20.0, limit=50.0, status="pass",
              util=40.0, expected=60.0, expected_breach=True, horizon="2026-12"),
        _test("soon", "Forecast soon", value=20.0, limit=50.0, status="pass",
              util=40.0, expected=60.0, expected_breach=True, horizon="2026-08"),
        _test("warn", "Warning now", value=47.0, limit=50.0, status="warning", util=94.0),
        _test("breach", "Breach now", value=60.0, limit=50.0, status="breach", util=120.0),
    ]))
    order = [r["label"] for r in C.select_tests(rows, limit=7)]
    assert order[0] == "Breach now"
    assert order[1] == "Warning now"
    # Both forecast breaches outrank the stress-only and the merely-high test,
    # and the EARLIER horizon comes first.
    assert order.index("Forecast soon") < order.index("Forecast late")
    assert order.index("Forecast late") < order.index("Stress only")
    assert order.index("Stress only") < order.index("High utilisation")
    assert order[-1] == "Low utilisation"


def test_configured_display_priority_breaks_ties():
    rows = C.adapt_tests(_env([
        _test("a", "Alpha", value=10.0, limit=50.0, status="pass", util=20.0,
              severity="low"),
        _test("b", "Bravo", value=10.0, limit=50.0, status="pass", util=20.0,
              severity="critical"),
    ]))
    assert [r["label"] for r in C.select_tests(rows)] == ["Bravo", "Alpha"]


def test_ranking_is_stable_on_a_complete_tie():
    """Identical tests must order identically on every run."""
    rows = C.adapt_tests(_env([
        _test("z", "Zulu", value=10.0, limit=50.0, status="pass", util=20.0),
        _test("a", "Alpha", value=10.0, limit=50.0, status="pass", util=20.0)]))
    first = [r["label"] for r in C.select_tests(rows)]
    assert first == [r["label"] for r in C.select_tests(rows)]
    assert first == ["Alpha", "Zulu"]          # stable test-id tie-break


def test_a_minimum_direction_limit_is_not_assumed_to_be_a_ceiling():
    """A 'min' test is a FLOOR. Status comes from the evaluator, and nothing
    here may invert it by assuming higher is worse."""
    rows = C.adapt_tests(_env([
        _test("floor", "Minimum liquidity", value=8.0, limit=10.0,
              status="breach", util=125.0, operator="min")]))
    row = rows[0]
    assert row["operator"] == "min"
    assert row["status"] == C.STATUS_BREACH
    assert C.select_tests(rows)[0]["label"] == "Minimum liquidity"


@pytest.mark.parametrize("unit,value,expected", [
    ("percent", 45.6, "45.6%"),
    ("gbp", 12_500_000.0, "£12.5m"),
    ("count", 1420.0, "1,420"),
    ("ratio", 2.35, "2.4"),
])
def test_values_render_in_their_configured_unit(unit, value, expected):
    assert C.format_measure(value, unit) == expected


def test_missing_forecast_and_stress_are_simply_absent():
    """A deployment that evaluates neither must not show invented values."""
    env = _env([_test("t", "Only current", value=10.0, limit=50.0,
                      status="pass", util=20.0)], forward=False)
    row = C.adapt_tests(env)[0]
    assert row["expected_value"] is None and row["stress_value"] is None
    assert C.forward_states_available(env) is False


def test_legacy_source_is_disclosed_as_unapproved():
    env = _env([_test("t", "T", value=1.0, limit=2.0, status="pass", util=50.0)],
               source=C.SOURCE_LEGACY)
    assert "not an operator-approved" in C.source_disclosure(env)


def test_summary_counts_come_from_the_rows_when_absent():
    rows = C.adapt_tests(_env([
        _test("b", "B", value=60.0, limit=50.0, status="breach", util=120.0),
        _test("w", "W", value=47.0, limit=50.0, status="warning", util=94.0),
        _test("s", "S", value=20.0, limit=50.0, status="pass", util=40.0,
              stress=60.0, stress_breach=True)]))
    summary = C.summarise(_env([]), rows)
    assert summary["breaches"] == 1
    assert summary["warnings"] == 1
    assert summary["stress_breaches"] == 1


# --------------------------------------------------------------------------- #
# Narrative.
# --------------------------------------------------------------------------- #

def _takeaway(rows, forward=True):
    from mi_agent_pptx.deck import DeckBuilder
    summary = C.summarise(_env([]), rows)
    return DeckBuilder._concentration_takeaway(
        object.__new__(DeckBuilder), summary, C.select_tests(rows), forward)


def test_narrative_reports_only_states_that_exist():
    rows = C.adapt_tests(_env([
        _test("a", "A", value=10.0, limit=50.0, status="pass", util=20.0)]))
    text = _takeaway(rows, forward=False)
    assert "All current tests remain within limit." in text
    assert "Forward-looking states were not evaluated" in text
    assert "stress" not in text.lower().replace("stress rather", "")


def test_narrative_separates_forecast_from_stress():
    rows = C.adapt_tests(_env([
        _test("f", "F", value=20.0, limit=50.0, status="pass", util=40.0,
              expected=60.0, expected_breach=True, horizon="2026-08"),
        _test("s", "S", value=20.0, limit=50.0, status="pass", util=40.0,
              stress=60.0, stress_breach=True)]))
    text = _takeaway(rows)
    assert "expected to breach on the governed forecast" in text
    assert "crossing around 2026-08" in text
    assert "stress rather than the expected outcome" in text


def test_narrative_contains_no_client_specific_wording():
    rows = C.adapt_tests(_env([
        _test("t", "South West concentration", value=46.0, limit=50.0,
              status="warning", util=93.0)]))
    text = _takeaway(rows)
    # The conclusion describes the STATE, not the particular test's name.
    assert "South West" not in text
    assert "warning range" in text
