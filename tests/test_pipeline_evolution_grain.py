"""tests/test_pipeline_evolution_grain.py — the grain a series is reported at.

`_route_evolution` keyed EVERY single-metric series on the producer's monthly
`period` field. The pipeline producer publishes one governed observation per
weekly extract and carries a day-level `week` alongside a lossy `period` month,
so five distinct weekly extracts arrived as five rows all labelled `2026-05`,
under a chart this same function titles "by week".

That was not cosmetic. A series whose x-values are all identical carries no time
axis, and the downstream time-axis guard refused the whole answer: "a series over
time ... the answer that was produced carries no time axis; it reports a single
position and cannot show movement". Two of the three canonical pipeline trend
questions were refused outright by a defect in how their x-axis was labelled.

These tests are written against the five-week fixture's movement table, not
copied back from a run, and they fail in BOTH directions: keying everything on
`period` breaks the pipeline assertions, keying everything on `week` breaks the
funded ones. A rule that satisfies both is the rule.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FIXTURE = _ROOT / "tests" / "fixtures" / "pipeline_history_5w"

#: The fixture's five governed extracts, from its movement table.
#:
#: These are LIVE PIPELINE STOCK — cases still capable of becoming funded loans.
#: They were previously the whole extract, which counted a completed case (which
#: has funded, and is in the funded book) and a withdrawn case (which has gone
#: away) as pipeline. Read straight off the fixture's own Status column:
#:
#:     week          rows  live      live amt       all amt
#:     2026-05-01       6     6     2,300,000     2,300,000
#:     2026-05-08       7     7     2,800,000     2,800,000
#:     2026-05-15       8     8     3,600,000     3,600,000
#:     2026-05-22       8     5     2,400,000     3,600,000   2 Completed, 1 Withdrawn
#:     2026-05-29       8     5     2,400,000     3,600,000   2 Completed, 1 Withdrawn
#:
#: The first three weeks are unchanged because the fixture has no terminal case
#: until week four — which is what makes this fixture worth asserting against.
WEEKS = ("2026-05-01", "2026-05-08", "2026-05-15", "2026-05-22", "2026-05-29")
EXPECTED_AMOUNT = [2_300_000.0, 2_800_000.0, 3_600_000.0, 2_400_000.0, 2_400_000.0]
EXPECTED_COUNT = [6, 7, 8, 5, 5]

#: The funded side: two synthetic month-end runs, so a monthly series has two
#: points and a weekly one would have none.
FUNDED_RUNS = (("mi_2026_04", "2026-04-30", 60, 1.0),
               ("mi_2026_05", "2026-05-31", 70, 1.15))
FUNDED_PERIODS = ["2026-04", "2026-05"]


def _write_run(root: Path, run_id: str, reporting_date: str, n: int, scale: float) -> None:
    # Seeded from the run id's own characters, so the tape is identical in every
    # process — `hash()` is salted per interpreter and would make a before/after
    # comparison of funded values meaningless.
    rng = np.random.default_rng(sum(ord(c) for c in run_id))
    pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": (rng.uniform(120_000, 280_000, n) * scale).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": rng.choice(["London", "South East", "Scotland"], n),
        "reporting_date": [reporting_date] * n,
    }).to_csv(_mkdir(root / "client_001" / run_id / "output" / "central")
              / "18_central_lender_tape.csv", index=False)


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="module")
def ask():
    """A live /mi/query caller against the five-week pipeline fixture."""
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    for run_id, rdate, n, scale in FUNDED_RUNS:
        _write_run(out_root, run_id, rdate, n, scale)
    prev = {k: os.environ.get(k) for k in
            ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT",
             "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(_FIXTURE)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    client = TestClient(app)

    def _ask(question: str) -> dict:
        return client.post("/mi/query", json={
            "question": question, "portfolioId": "client_001/mi_2026_05",
            "asOfDate": "2026-05-31"}).json()

    yield _ask
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _series(resp: dict):
    charts = [a for a in (resp.get("artifacts") or []) if a.get("type") == "chart"]
    rows = charts[0].get("rows") if charts else []
    return ([r.get("period") for r in rows or []],
            [r.get("value") for r in rows or []])


# --------------------------------------------------------------------------- #
# The pipeline side: five weekly observations, five distinct labels
# --------------------------------------------------------------------------- #
def test_the_pipeline_trend_is_reported_week_by_week(ask):
    """The defect's headline victim. Five governed extracts, five x-values."""
    x, vals = _series(ask("How has the pipeline changed over time?"))
    assert x == list(WEEKS), x
    assert vals == EXPECTED_AMOUNT, vals


def test_the_pipeline_case_count_trend_is_reported_week_by_week(ask):
    x, vals = _series(ask("Show pipeline case count over time."))
    assert x == list(WEEKS), x
    assert vals == EXPECTED_COUNT, vals


def test_a_weekly_pipeline_trend_is_delivered_not_refused(ask):
    """Before the fix this returned ok=False. A collapsed x-axis is indistinguishable
    from no time axis, and the time-axis guard correctly refused it."""
    r = ask("How has the pipeline changed over time?")
    assert r.get("ok") is True, r.get("answer")
    assert not any("no time axis" in str(w) for w in (r.get("warnings") or []))
    assert "5 period(s)" in (r.get("answer") or "")


def test_the_pipeline_series_labels_are_all_distinct(ask):
    """The property the defect destroyed, stated on its own: a series with five
    observations must not collapse onto one point."""
    x, _ = _series(ask("Show the pipeline trend."))
    assert len(x) == 5 and len(set(x)) == 5, x


# --------------------------------------------------------------------------- #
# The funded side: unchanged, and unchangeable by the pipeline rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "Show funded balance evolution by month.",
    "Show loan count evolution by month.",
    "Show the funded balance trend over time.",
    "Show average LTV evolution by month.",
])
def test_funded_evolution_stays_monthly(ask, question):
    x, _ = _series(ask(question))
    assert x == FUNDED_PERIODS, (question, x)


def test_a_funded_series_carries_no_week_to_be_keyed_on():
    """Why the rule is safe: the funded producers publish no observation identity
    finer than the month, so there is nothing for a weekly rule to pick up. Read
    from the producer, not from the dataset name."""
    from mi_agent_api import evolution as evolution_mod
    import inspect
    import re

    def published_keys(fn):
        block = inspect.getsource(fn).split("periods.append(", 1)[1]
        depth, out = 0, []
        for ch in block:
            out.append(ch)
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        return set(re.findall(r'"([a-zA-Z_]+)":', "".join(out)))

    from mi_agent_api import chat_routing
    funded = published_keys(evolution_mod.assemble_funded_evolution)
    filtered = published_keys(chat_routing._filtered_funded_evo)
    pipeline = published_keys(evolution_mod.pipeline_evolution)
    assert "week" not in funded and "period" in funded
    assert "week" not in filtered and "period" in filtered
    assert "week" in pipeline and "period" in pipeline


def test_the_route_reads_the_grain_from_the_series_not_the_dataset_name():
    """The rule must not be a dataset allowlist. If it ever becomes one this
    fails, because the source no longer keys the decision on what was returned."""
    import inspect
    import re
    from mi_agent_api import chat_routing

    src = inspect.getsource(chat_routing._route_evolution)
    # Assignments only — `period_field == "week"` in the grain DECLARATION reads
    # the decision, it does not make a second one.
    decision = [ln for ln in src.splitlines()
                if re.match(r"\s*period_field\s*=[^=]", ln)]
    assert len(decision) == 1, decision
    assert 'for p in periods' in decision[0], decision[0]
    assert 'dataset' not in decision[0], decision[0]


# --------------------------------------------------------------------------- #
# The SECOND owner: what the receipt believes the grain is
# --------------------------------------------------------------------------- #
# `execution_receipt._ROUTE_TIME_GRAIN` asserted "month" for all ten series
# routes. Three of them publish weekly. While the series was wrongly keyed
# monthly the two claims agreed by accident; keying it correctly made them
# disagree in BOTH directions at once — a weekly question refused as monthly,
# and a monthly question answered weekly with nothing disclosed. The route now
# declares what it published and the receipt reads that declaration.
@pytest.mark.parametrize("question,delivers,says", [
    ("Show pipeline amount evolution by week.", True, None),
    ("Show pipeline amount evolution by month.", False,
     "reported at week level, not by month"),
    ("Show funded balance evolution by month.", True, None),
    ("Show funded balance evolution by week.", False,
     "reported at month level, not by week"),
])
def test_every_grain_quadrant_is_answered_or_stated(ask, question, delivers, says):
    """All four combinations of {funded, pipeline} x {month, week}. A wrong
    answer here is either a false refusal or an UNDISCLOSED substitution, and
    the two failure modes sit on opposite sides of this table."""
    r = ask(question)
    assert r.get("ok") is delivers, (question, r.get("answer"))
    if says:
        assert says in (r.get("answer") or ""), (question, r.get("answer"))


def test_a_weekly_funnel_question_is_no_longer_told_it_is_monthly(ask):
    """`evolution_funnel` has ALWAYS keyed its rows on `week` — this was wrong
    before any change to the single-metric series, and is the proof the stale
    claim was pre-existing rather than introduced."""
    r = ask("Show the KFI trend by week.")
    assert r.get("ok") is True, r.get("answer")
    assert "not by week" not in (r.get("answer") or "")


def test_a_route_that_declares_nothing_keeps_the_static_fallback():
    """The blast is bounded: only routes that declare a grain are affected, so
    the other seven entries in the map behave exactly as before."""
    from mi_agent import execution_receipt as R

    assert R.declared_series_grain(None) is None
    assert R.declared_series_grain({}) is None
    assert R.declared_series_grain({"metadata": {}}) is None
    assert R.declared_series_grain({"metadata": {"seriesGrain": "week"}}) == "week"

    undeclared = R.time_axis_disclosure("week", "temporal_compare", {"metadata": {}})
    assert undeclared is not None and undeclared.concepts == ("week", "month")
    declared = R.time_axis_disclosure("week", "temporal_compare",
                                      {"metadata": {"seriesGrain": "week"}})
    assert declared is not None and declared.concepts == ("week", "week")


def test_a_question_naming_no_unit_raises_no_grain_facet():
    """Unchanged and load-bearing: a point-in-time KPI must never be told it
    failed to honour a grain nobody asked for."""
    from mi_agent import execution_receipt as R

    assert R.time_axis_disclosure(None, "evolution",
                                  {"metadata": {"seriesGrain": "week"}}) is None
    assert R.time_axis_disclosure("", "evolution", None) is None


def test_the_declaration_is_execution_evidence_not_prose():
    """The rule this fix must not break: grain is read from what the route
    reports, never from the answer sentence."""
    import inspect
    from mi_agent import execution_receipt as R

    src = inspect.getsource(R.declared_series_grain)
    assert "metadata" in src
    for prose_field in ("answer", "question", "interpreted"):
        assert f'get("{prose_field}")' not in src, prose_field


def test_a_monthly_question_on_the_weekly_funnel_is_told_so(ask):
    """The case that showed the pre-registration was mis-specified.

    `resolve_dataset("completions by month")` says FUNDED — the wording carries
    no pipeline term — but the route that answers is the pipeline funnel, whose
    series carries a `week` key and no `period` key at all. So the dataset LABEL
    and the producer's GRAIN are independent facts, and a pre-registration that
    said "funded questions must not move" was describing the wrong axis.

    Before the receipt read the route's declaration, these two delivered weekly
    numbers to a monthly question and disclosed nothing. Refusing them is the
    same correction as the pipeline `by month` case, not a regression.
    """
    for q in ("completions by month", "Show expected completions by month."):
        r = ask(q)
        assert r.get("ok") is False, (q, r.get("answer"))
        assert "reported at week level, not by month" in (r.get("answer") or ""), q


def test_the_funnel_series_publishes_only_a_week():
    """The producer-side fact the test above rests on."""
    from mi_agent_api import evolution as evolution_mod

    fixture = str(_FIXTURE)
    funnel = evolution_mod.pipeline_funnel_evolution(fixture, "client_001", "mi_2026_05")
    points = funnel.get("series", {}).get("COMPLETED", [])
    assert points, "the five-week fixture must reach the funnel"
    assert set(points[0]) == {"week", "value", "count"}
    assert [p["week"] for p in points] == list(WEEKS)
