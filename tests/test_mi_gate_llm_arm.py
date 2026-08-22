"""tests/test_mi_gate_llm_arm.py — the Gate instrument must be able to FAIL.

No instrument ships without a test proving it can fail. Every assertion here
feeds the instrument a case it is supposed to CATCH and proves it catches it —
and, just as importantly, a neighbouring case it is supposed to PASS, so the
instrument is not merely a function that always says "bad".

The three failure modes are the three this programme has actually hit:

* rating from the RECEIPT instead of the artifact (a truthful-looking
  ``dimensionsApplied`` over a table carrying one series);
* concluding the model ran from a FIELD VALUE rather than a counted call;
* averaging away self-disagreement across repeats.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Any, Dict, List

import pytest

from question_interpretation import mi_gate_llm_arm as G
from question_interpretation.time_series_surface import ABSENT, PROVEN, SHAPES


# --------------------------------------------------------------------------- #
# helpers — build a response the way mi_service would
# --------------------------------------------------------------------------- #
def _response(*, ok: bool, rows: List[Dict[str, Any]],
              claimed_dims=(), guard: Dict[str, Any] | None = None,
              not_applied=None, route: str = "evolution") -> Dict[str, Any]:
    return {
        "ok": ok,
        "artifacts": [{"type": "table", "rows": rows}],
        "metadata": {"route": route, "parserMode": None,
                     "semantic_guard": guard or {}},
        "executionSummary": {"dimensionsApplied": list(claimed_dims),
                             "notApplied": not_applied,
                             "measure": "balance", "population": "funded"},
        "answer": "",
    }


def _observe_once(response: Dict[str, Any], want: List[str],
                  calls: int = 0) -> Dict[str, Any]:
    """Run ``observe`` against a canned response, simulating ``calls`` invocations."""
    counter = G.InvocationCounter()

    def ask(_q):
        counter.n += calls
        return response

    return G.observe(ask, "q", want, counter)


_SERIES_ONLY = [{"period": "2025-01", "balance": 10},
                {"period": "2025-02", "balance": 11},
                {"period": "2025-03", "balance": 12}]

_SERIES_BY_REGION = [{"period": "2025-01", "region": "North", "balance": 5},
                     {"period": "2025-01", "region": "South", "balance": 5},
                     {"period": "2025-02", "region": "North", "balance": 6},
                     {"period": "2025-02", "region": "South", "balance": 6}]


# --------------------------------------------------------------------------- #
# 1. THE RECEIPT MUST NOT BE ABLE TO BUY A PASS
# --------------------------------------------------------------------------- #
def test_false_receipt_is_caught_as_absent_and_silent():
    """ok=True + a receipt claiming `region` + an artifact carrying ONE series.

    This is the exact shape the surface exists to catch. If the instrument
    rated from the receipt it would call this PROVEN.
    """
    seen = _observe_once(
        _response(ok=True, rows=_SERIES_ONLY, claimed_dims=["region"]),
        want=["region"])
    assert seen["rating"] == ABSENT, "a whole-book series for a segmented request is ABSENT"
    assert "WHOLE-BOOK SERIES" in seen["why"]
    assert "region" in seen["why"], "the false claim must be NAMED in the reason"
    assert seen["silent_drop"] is True, "ok=True with nothing disclosed is a SILENT drop"


def test_the_same_shape_disclosed_is_not_a_silent_drop():
    """The neighbouring case: same loss, but DISCLOSED. Not silent."""
    seen = _observe_once(
        _response(ok=True, rows=_SERIES_ONLY, claimed_dims=["region"],
                  guard={"verdict": "partial"}),
        want=["region"])
    assert seen["rating"] == ABSENT, "still ABSENT — disclosure does not restore the axis"
    assert seen["silent_drop"] is False, "a named loss is honest, not silent"


def test_a_real_breakdown_over_time_is_proven():
    """The instrument must still PASS a genuine result, or it proves nothing."""
    seen = _observe_once(
        _response(ok=True, rows=_SERIES_BY_REGION, claimed_dims=["region"]),
        want=["region"])
    assert seen["rating"] == PROVEN
    assert seen["silent_drop"] is False


def test_breakdown_with_no_time_axis_is_absent():
    """A breakdown delivered over ONE period is a dropped time axis."""
    one_period = [{"period": "2025-01", "region": "North", "balance": 5},
                  {"period": "2025-01", "region": "South", "balance": 6}]
    seen = _observe_once(_response(ok=True, rows=one_period,
                                   claimed_dims=["region"]), want=["region"])
    assert seen["rating"] == ABSENT
    assert "TIME AXIS DROPPED" in seen["why"]


# --------------------------------------------------------------------------- #
# 2. THE MODEL MUST BE COUNTED, NOT INFERRED
# --------------------------------------------------------------------------- #
def test_model_calls_are_counted_not_read_from_the_field():
    """parserMode is None in both cases; only the COUNT distinguishes them."""
    ran = _observe_once(_response(ok=True, rows=_SERIES_ONLY), want=[], calls=2)
    didnt = _observe_once(_response(ok=True, rows=_SERIES_ONLY), want=[], calls=0)
    assert ran["parser_mode"] == didnt["parser_mode"] is None, \
        "the field cannot tell these apart — that is the point"
    assert ran["model_calls"] == 2
    assert didnt["model_calls"] == 0


def test_counter_wraps_the_real_invoke_and_is_idempotent():
    """install() must actually wrap `_invoke`, and must not stack wrappers."""
    from mi_agent import llm_query_parser as P
    original = P._invoke
    counter = G.InvocationCounter()
    try:
        counter.install()
        counter.install()  # second install must be a no-op, not a second wrapper
        assert P._invoke is not original, "install() did not wrap _invoke"
        P._invoke(prompt="p", model="m", llm_callable=lambda _p: "text")
        assert counter.n == 1, "a double-installed wrapper would count 2"
    finally:
        counter.uninstall()
    assert P._invoke is original, "uninstall() must restore the original"


def test_a_run_with_no_calls_is_reported_void_not_identical():
    """Zero counted calls across the whole run => the comparison is VOID.

    This is the guard against the earlier "15 of 15 identical" report.
    """
    row = {"shape": "T1", "shape_name": "metric x time", "question": "q",
           "deterministic": _observe_once(_response(ok=True, rows=_SERIES_ONLY),
                                          want=[]),
           "llm_modal": _observe_once(_response(ok=True, rows=_SERIES_ONLY),
                                      want=[]),
           "llm_runs": [], "distinct_outcomes": 1, "modal_share": "5/5",
           "stable": True, "model_calls": [0], "parser_modes": [None],
           "attribution": G.SAME}
    result = {"book": "alderbridge", "repeats": 5, "llm_available": True,
              "rows": [row], "deterministic": {}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = G.report(result)
    assert rc == 3, "a zero-call run must not return success"
    assert "MADE NO CALLS" in buf.getvalue()
    assert "void" in buf.getvalue().lower()


# --------------------------------------------------------------------------- #
# 3. SELF-DISAGREEMENT MUST NOT BE AVERAGED AWAY
# --------------------------------------------------------------------------- #
def test_outcome_key_separates_runs_that_differ_in_the_artifact():
    proven = _observe_once(_response(ok=True, rows=_SERIES_BY_REGION,
                                     claimed_dims=["region"]), want=["region"])
    absent = _observe_once(_response(ok=True, rows=_SERIES_ONLY,
                                     claimed_dims=["region"]), want=["region"])
    assert G.outcome_key(proven) != G.outcome_key(absent), \
        "two different artifacts must not collapse into one outcome"


def test_outcome_key_ignores_prose_only_differences():
    a = _response(ok=True, rows=_SERIES_ONLY)
    b = _response(ok=True, rows=_SERIES_ONLY)
    b["answer"] = "a completely different sentence"
    assert G.outcome_key(_observe_once(a, want=[])) == \
           G.outcome_key(_observe_once(b, want=[])), \
        "differing prose over an identical artifact is the SAME outcome"


# --------------------------------------------------------------------------- #
# 4. ATTRIBUTION MUST NOT FORCE A CASE INTO A BOX THAT DOES NOT FIT
# --------------------------------------------------------------------------- #
def test_guard_refused_a_proposal_the_route_would_have_handled():
    det = _observe_once(_response(ok=True, rows=_SERIES_ONLY), want=[])
    llm = _observe_once(_response(ok=False, rows=[]), want=[])
    assert G.attribute(det, llm) == G.GUARD_REFUSED_PROPOSAL


def test_model_proposed_something_the_route_does_not_reach():
    det = _observe_once(_response(ok=True, rows=_SERIES_BY_REGION,
                                  claimed_dims=["region"]), want=["region"])
    llm = _observe_once(_response(ok=True, rows=_SERIES_ONLY,
                                  claimed_dims=["region"]), want=["region"])
    assert G.attribute(det, llm) == G.PROPOSAL_NOT_REACHED


def test_llm_answering_where_deterministic_refused_fits_neither_box():
    det = _observe_once(_response(ok=False, rows=[]), want=[])
    llm = _observe_once(_response(ok=True, rows=_SERIES_ONLY), want=[])
    assert G.attribute(det, llm) == G.UNCLASSIFIED, \
        "the brief says SAY SO rather than force it into one of the two"


def test_identical_outcomes_attribute_as_same():
    det = _observe_once(_response(ok=True, rows=_SERIES_ONLY), want=[])
    llm = _observe_once(_response(ok=True, rows=_SERIES_ONLY), want=[])
    assert G.attribute(det, llm) == G.SAME


# --------------------------------------------------------------------------- #
# 5. THE P0 REFUSALS MUST BE REPORTABLE AS BREACHED
# --------------------------------------------------------------------------- #
def _p0_result(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    question = G.P0_REFUSALS[0][0]
    return {"book": "alderbridge", "repeats": len(runs), "llm_available": True,
            "rows": [{"shape": "T5", "shape_name": "x", "question": question,
                      "deterministic": runs[0], "llm_modal": runs[0],
                      "llm_runs": runs, "distinct_outcomes": 1,
                      "modal_share": "1/1", "stable": True,
                      "model_calls": [1], "parser_modes": [None],
                      "attribution": G.SAME}],
            "deterministic": {}}


def test_p0_reported_as_held_when_every_repeat_refuses():
    refused = _observe_once(_response(ok=False, rows=[]), want=["region"])
    status = G.p0_status(_p0_result([refused, refused]))[0]
    assert status["holds"] is True
    assert "held" in status["note"]


def test_p0_reported_as_breached_on_a_single_silent_drop():
    """One silent drop in five repeats BREACHES the refusal. It is not a minority
    outcome to be averaged away."""
    refused = _observe_once(_response(ok=False, rows=[]), want=["region"])
    silent = _observe_once(_response(ok=True, rows=_SERIES_ONLY,
                                     claimed_dims=["region"]), want=["region"])
    assert silent["silent_drop"] is True
    status = G.p0_status(_p0_result([refused, refused, silent]))[0]
    assert status["holds"] is False
    assert "BREACHED" in status["note"]
    assert status["silent_drops"] == 1


# --------------------------------------------------------------------------- #
# 6. THE PHRASINGS COME FROM THE DECLARATION, NOT FROM A COPY
# --------------------------------------------------------------------------- #
def test_phrasings_are_read_from_shapes_and_number_29():
    got = G.phrasings()
    assert len(got) == 29, "the surface declares 29 phrasings"
    declared = [(sid, name, q) for sid, name, qs, _w in SHAPES for q in qs]
    assert [(s, n, q) for s, n, q, _w in got] == declared, \
        "phrasings() must mirror SHAPES exactly — never a retyped copy"


def test_named_p0_questions_exist_in_the_declared_phrasings():
    declared = {q for _s, _n, qs, _w in SHAPES for q in qs}
    for question, _limb in G.P0_REFUSALS:
        assert question in declared, \
            "a P0 question the surface does not run cannot be reported on"


# --------------------------------------------------------------------------- #
# 7. THE RUN TOTAL MUST BE THE RUN TOTAL
# --------------------------------------------------------------------------- #
def test_counter_accumulates_across_observations():
    """Regression: `observe` used to reset the counter, so the run total field
    reported only the LAST question's calls (1) while the real total was 130.

    Per-question counts must still be per-question, and the counter must still
    be cumulative across the run.
    """
    counter = G.InvocationCounter()
    response = _response(ok=True, rows=_SERIES_ONLY)

    def ask_twice(_q):
        counter.n += 2
        return response

    first = G.observe(ask_twice, "q1", [], counter)
    second = G.observe(ask_twice, "q2", [], counter)
    assert first["model_calls"] == 2, "per-question count must be the delta"
    assert second["model_calls"] == 2, "second question must not inherit the first"
    assert counter.n == 4, "the counter must accumulate the RUN total"


# --------------------------------------------------------------------------- #
# 8. THE BRIEF'S TIME-AXIS RULE, IN FULL
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cols,expected", [
    (["measure", "population", "period", "prior", "current", "change"], ("prior", "current")),
    (["rank", "category", "start_value", "end_value", "movement"], ("start_value", "end_value")),
    (["opening_balance", "closing_balance"], ("opening_balance", "closing_balance")),
    (["previous_period", "latest_period"], ("previous_period", "latest_period")),
    (["period", "balance"], None),
    (["region", "balance"], None),
])
def test_movement_pair_detects_the_two_ends_the_brief_names(cols, expected):
    assert G.movement_pair(cols) == expected


def test_movement_pair_does_not_fire_on_one_end_alone():
    assert G.movement_pair(["start_value", "rank"]) is None, \
        "one end of a movement is not a movement"


def test_brief_rule_credits_a_movement_pair_the_surface_misses():
    """The exact T8 artifact: `period` carries ONE value, but prior/current is a
    time axis under the brief's rule. The surface rates this ABSENT; the brief's
    rule must not."""
    rows = [{"measure": "balance", "population": "front", "period": "2025-03",
             "prior": 10, "current": 12, "change": 2},
            {"measure": "balance", "population": "back", "period": "2025-03",
             "prior": 20, "current": 19, "change": -1}]
    response = _response(ok=True, rows=rows)
    seen = _observe_once(response, want=[])
    assert seen["rating"] == ABSENT, "the inherited surface rule misses the pair"
    assert seen["rating_brief"] == PROVEN, "the brief's rule credits the pair"
    assert "movement pair" in seen["why_brief"]


def test_brief_rule_still_refuses_a_segmented_request_with_no_requested_dimension():
    """A movement pair does NOT buy a pass on the breakdown. Time axis present,
    requested dimension absent, still ABSENT — and the columns that DO carry
    values are named rather than hidden."""
    rows = [{"rank": 1, "category": "North", "start_value": 1, "end_value": 2},
            {"rank": 2, "category": "South", "start_value": 3, "end_value": 4}]
    seen = _observe_once(_response(ok=True, rows=rows), want=["region"])
    assert seen["rating_brief"] == ABSENT
    assert "time axis present" in seen["why_brief"]
    assert "category" in seen["why_brief"], \
        "a generically named column must be NAMED, not silently scored absent"


def test_brief_rule_does_not_credit_a_refusal():
    seen = _observe_once(_response(ok=False, rows=[]), want=["region"])
    assert seen["rating_brief"] == ABSENT
    assert seen["why_brief"] == "refused"


def test_column_census_records_every_column_not_just_matched_ones():
    rows = [{"category": "North", "balance": 1}, {"category": "South", "balance": 2}]
    census = G.column_census([{"type": "table", "rows": rows}])
    assert set(census["columns"]) == {"category", "balance"}
    assert census["columns"]["category"]["distinct"] == 2
    assert census["columns"]["category"]["sample"] == ["North", "South"]
