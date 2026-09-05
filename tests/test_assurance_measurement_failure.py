"""tests/test_assurance_measurement_failure.py — crashed measurement is not zero.

The P0 loader audit closed one shape of silent assurance failure. Four
instruments carried another: a broad `except` that turned a crashed MEASUREMENT
into an empty result, so the run continued and reported a number.

Reproduced before remediation, each one exited 0 and looked healthy:

    contract_role_census          645/645 questions faulted -> "questions
                                  compared: 645, ILLEGAL deltas (blast): 0"
    equivalence_portfolio_summary registry faulted -> "economic differences: 0"
                                  over nine cases, on the pre-1G reading the
                                  file's own comment calls meaningless
    filter_ownership_trace        every corpus parse faulted -> "corpus questions
                                  carrying spec.filters: 0", and the headline
                                  "expressible by lens_filters: 0" was IDENTICAL
                                  to the real finding
    route_ownership_evolution     every query faulted -> "OWNED BY THE EVOLUTION
                                  FAMILY: 0" and a table of zeros

The distinction these tests pin is the whole point: a measurement that ran and
found nothing is evidence; a measurement that could not run is not, and the two
must not share a representation.
"""
from __future__ import annotations

import io
import contextlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.assurance_semantics import (  # noqa: E402
    AssuranceError, AssuranceMeasurementError, AssuranceSemanticsError,
    measurement_failed)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #
def test_both_assurance_failures_share_one_base():
    """So a caller can catch 'this run cannot be trusted' in one place."""
    assert issubclass(AssuranceSemanticsError, AssuranceError)
    assert issubclass(AssuranceMeasurementError, AssuranceError)
    assert not issubclass(AssuranceMeasurementError, AssuranceSemanticsError)


def test_the_failure_names_instrument_case_and_cause():
    exc = measurement_failed("some_instrument", "a question", ValueError("boom"))
    text = str(exc)
    assert "ASSURANCE INVALID" in text
    assert "some_instrument" in text and "a question" in text
    assert "ValueError" in text and "boom" in text


def test_the_root_cause_is_not_swallowed():
    """`raise ... from exc` at every site, so the original traceback survives."""
    import ast
    import inspect

    for mod in ("contract_role_census", "equivalence_portfolio_summary",
                "filter_ownership_trace", "route_ownership_evolution"):
        src = (_REPO / "migration_phase0" / f"{mod}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        raises = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Raise)
                  and "measurement_failed" in ast.dump(n)]
        assert raises, mod
        for node in raises:
            assert node.cause is not None, f"{mod}: raise without `from exc`"


# --------------------------------------------------------------------------- #
# SITE 1 — contract_role_census
# --------------------------------------------------------------------------- #
def test_site1_a_parse_fault_fails_loudly(monkeypatch):
    import mi_agent.llm_query_parser as lqp

    import migration_phase0.contract_role_census as crc

    monkeypatch.setattr(lqp, "parse_with_repair",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(AssuranceMeasurementError) as exc:
        _quiet(crc._snapshot, tempfile.mktemp(suffix=".json"))
    assert "contract_role_census" in str(exc.value)


def test_site1_legitimate_zero_dimension_rows_are_still_recorded():
    """275 of 645 questions genuinely name no dimension. The fix must not turn
    'nothing matched' into an error."""
    import migration_phase0.contract_role_census as crc

    out = tempfile.mktemp(suffix=".json")
    _quiet(crc._snapshot, out)
    rows = json.loads(Path(out).read_text())
    assert len(rows) == 645
    assert not any("error" in r for r in rows)
    assert sum(1 for r in rows if not r.get("dimensions")) > 100
    assert sum(1 for r in rows if not r.get("filters")) > 100


def test_site1_diff_refuses_a_census_carrying_error_rows(tmp_path):
    """An older census file could still hold error rows, and `_diff` read those
    through `.get("dimensions", [])` as zero dimensions."""
    import migration_phase0.contract_role_census as crc

    bad = [{"question": "q1", "error": "RuntimeError: injected"}]
    b = tmp_path / "b.json"
    a = tmp_path / "a.json"
    b.write_text(json.dumps(bad))
    a.write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as exc:
        _quiet(crc._diff, str(b), str(a))
    assert "measurement error" in str(exc.value)


# --------------------------------------------------------------------------- #
# SITE 2 — equivalence_portfolio_summary
# --------------------------------------------------------------------------- #
def test_site2_a_registry_fault_fails_loudly(monkeypatch):
    import mi_agent_api.portfolio_context as ctx

    import migration_phase0.equivalence_portfolio_summary as eq

    monkeypatch.setattr(ctx, "build_registry",
                        lambda df: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(AssuranceMeasurementError) as exc:
        _quiet(eq.main)
    assert "equivalence_portfolio_summary" in str(exc.value)


# --------------------------------------------------------------------------- #
# SITE 3 — filter_ownership_trace
# --------------------------------------------------------------------------- #
def test_site3_a_corpus_parse_fault_fails_loudly(monkeypatch):
    import mi_agent.parsed_question as pq

    import migration_phase0.filter_ownership_trace as fot

    original = pq.ParsedQuestion.parse
    calls = {"n": 0}

    def flaky(q, sem):
        calls["n"] += 1
        if calls["n"] > 6:            # let the probes through, fail the corpus
            raise RuntimeError("injected")
        return original(q, sem)

    monkeypatch.setattr(pq.ParsedQuestion, "parse", staticmethod(flaky))
    with pytest.raises(AssuranceMeasurementError) as exc:
        _quiet(fot.main)
    assert "filter_ownership_trace" in str(exc.value)


def test_site3_reports_the_denominator_it_examined():
    """"0 filtered of 0 examined" and "0 of 882" printed the same headline."""
    import migration_phase0.filter_ownership_trace as fot

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fot.main()
    text = buf.getvalue()
    assert "corpus questions examined             : 882" in text
    # 119 until the place-resolver fallback stopped inventing a geography for
    # any phrase it was handed with no value catalogue. The four it loses are
    # `collateral_geography` = 'Equity Release Supermarket Limited' (twice, a
    # BROKER), 'October' (a MONTH) and 'Weighting' (an analytic noun). None of
    # the four is a place, none reached a reader — this trace parses without a
    # frame, and serving always has one — and each is now recorded as an
    # unresolved narrowing instead of a filter on a field nobody named.
    #
    # 115 -> 116 when the postfix comparators stopped carrying their own number
    # grammar. The ONE question that moved, across all 882:
    #
    #     "Drill into the 50%+ LTV bucket."
    #         {}  ->  {current_loan_to_value: {op: ge, value: 50.0}}
    #
    # `50%+` is a bound the prefix grammar would have read and the postfix
    # patterns could not, because theirs had no `%`. This census is the
    # measurement that says the change reached one corpus question and no
    # others — which is why the number is asserted rather than computed.
    assert "corpus questions carrying spec.filters: 116" in text


# --------------------------------------------------------------------------- #
# SITE 4 — route_ownership_evolution
# --------------------------------------------------------------------------- #
def test_site4_a_routing_fault_fails_loudly(monkeypatch):
    import fastapi.testclient as tc

    import migration_phase0.route_ownership_evolution as roe

    monkeypatch.setattr(tc.TestClient, "post",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(AssuranceMeasurementError) as exc:
        _quiet(roe.main, [])
    assert "route_ownership_evolution" in str(exc.value)


def test_site4_a_refused_answer_is_a_legitimate_reading():
    """A REFUSED grade is a measurement, not a failure — the owned questions
    that refuse must keep counting toward the denominator.

    The owned COUNT is a measurement of today's routing, not an invariant, and
    it moves when a question changes hands. It went 34 -> 35 when the
    filtered-summary branch stopped claiming questions that name a breakdown:
    "Show monthly loan count evolution by broker." had been claimed by that
    branch and graded unmapped, and now reaches the evolution route this file
    measures. The assertion is kept exact rather than loosened to ">=" so that a
    silent drift in the other direction still fails here.
    """
    import migration_phase0.route_ownership_evolution as roe

    rows = _quiet(roe.run)
    assert len(rows) == 882
    owned = [r for r in rows if r.get("owned")]
    assert len(owned) == 35
    assert sum(1 for r in owned if r["grade"] == "REFUSED") > 0
    assert sum(1 for r in owned if r["grade"] == "DELIVERED") > 0
    assert not any("error" in r for r in rows)
