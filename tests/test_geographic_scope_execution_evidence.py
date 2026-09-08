"""A geographic scope is APPLIED only when execution proves the PLACE ran.

THE DEFECT THIS PINS. `KIND_GEOGRAPHIC_SCOPE`'s identity is a VALUE — the facet
for "Show funded balance over time for Scotland" is labelled `Scotland` — but
the only evidence a routed answer carried was field-named: the executor's
`applied_filter_fields` and the route's `populationApplied.applied` prose both
say WHICH FIELD narrowed and neither says which value it compared. So the
routed reconciler had nothing to match a place against and refused every
geographic filter on a trend, while the same question without "over time"
answered.

`mi_query_executor.predicate_evidence` now records what was executed — field,
canonical field, operator, values — and `execution_receipt.geographic_scope_
executed` proves the facet from it. The tests below are written to FALSIFY that
proof as much as to confirm it: the value must match, the field must be one the
request resolves to, and a spelling nobody has governed as equivalent must not
pass.

ROW COUNTS ARE NOT THE CRITERION, and two tests say so in both directions. A
predicate that ran correctly can leave the count unmoved (every loan in a
single-region book is in that region) and can equally leave nothing. Both were
applied. Successful execution is the evidence; the counts are audit context.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as receipt
from mi_agent.execution_receipt import KIND_GEOGRAPHIC_SCOPE, RequestedFacet
from mi_agent_api.app import app

_PIPELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"

#: Deliberately DIFFERENT per basis, so a test that confuses collateral with
#: borrower geography fails instead of passing by coincidence.
_COLLATERAL = ("Scotland", "London", "South East", "Wales")
_OBLIGOR = ("North West", "Yorkshire and The Humber", "East Midlands", "Scotland")

#: Fixed per run, so the book is the same book on every machine and every run.
_SEEDS = {"mi_2025_10": 1_000, "mi_2025_11": 1_001, "mi_2025_12": 1_002}


def _write_run(root: Path, run_id: str, reporting_date: str, n: int,
               scale: float) -> None:
    # Seeded from the run's POSITION, not from `hash(run_id)`: Python salts
    # string hashing per process, so a hash-seeded fixture is a different book
    # on every run and a failure cannot be reproduced from the test alone.
    rng = np.random.default_rng(_SEEDS[run_id])
    frame = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": (rng.uniform(120_000, 280_000, n)
                                        * scale).round(2),
        "current_loan_to_value": rng.uniform(20, 75, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta"], n),
        "collateral_geography": rng.choice(_COLLATERAL, n),
        "geographic_region_obligor": rng.choice(_OBLIGOR, n),
        # Seasoning is DERIVED from origination against the reporting date, so
        # a book without it cannot express "the front book" — and the test that
        # non-geographic populations are unchanged would pass for the wrong
        # reason. Spread either side of the governed front-book boundary.
        "origination_date": pd.to_datetime(reporting_date)
        - pd.to_timedelta(rng.integers(30, 2_600, n), unit="D"),
        "reporting_date": [reporting_date] * n,
    })
    out = root / "client_001" / run_id / "output" / "central"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "18_central_lender_tape.csv", index=False)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(_REPO_ROOT)
    root = tmp_path / "onboarding_output"
    _write_run(root, "mi_2025_10", "2025-10-31", 90, 1.00)
    _write_run(root, "mi_2025_11", "2025-11-30", 95, 1.10)
    _write_run(root, "mi_2025_12", "2025-12-31", 99, 1.20)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(_PIPELINE_FIXTURE))
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    # THE SAME RUN ON BOTH PATHS. Without a portfolio registry this bare book
    # leaves run selection to discovery, and the current-period path settles on
    # the FIRST run while the series ends at the last — a pre-existing
    # difference (it reproduces unchanged on main) that has nothing to do with
    # what these tests measure. Naming the run removes it, so a parity failure
    # here means the two paths disagreed about the SAME frame.
    monkeypatch.setenv("MI_AGENT_RUN_ID", "mi_2025_12")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # THIS BOOK, AND NO OTHER. The MI read path can be pointed at a platform
    # blob root instead of an onboarding output root, and a module-scoped
    # fixture elsewhere in the suite that selects a different book by setting
    # these leaves them set. Run alone the file passed; run after such a test
    # the four data-dependent cases answered over someone else's portfolio.
    # Cleared rather than assumed, so the fixture states its own book
    # completely and the file's result does not depend on collection order.
    for leaked in ("MI_AGENT_PLATFORM_URI", "TRAKT_LOCAL_BLOB_ROOT",
                   "TRAKT_STORAGE_BACKEND", "TRAKT_PORTFOLIO_REGISTRY",
                   "MI_AGENT_CLIENT_ID", "MI_AGENT_REPORTING_DATE",
                   "MI_AGENT_LLM_ENABLED"):
        monkeypatch.delenv(leaked, raising=False)
    # AND THE CACHES THE ENVIRONMENT NO LONGER DESCRIBES. Clearing the variables
    # is not enough: the active dataset, the governed portfolio registry derived
    # from it, and the semantics registry are all process-global and outlive the
    # test that populated them. A neighbour that served a different book leaves
    # this file answering over that book with a correct-looking envelope, which
    # is how three of these cases failed inside the suite and passed alone.
    # Reset on the way IN so this file states its own world, and on the way OUT
    # so it does not become the neighbour that breaks the next one.
    def _reset_caches():
        from mi_agent_api import data_source
        from mi_workflows import semantics as _semantics
        data_source.reset_cache()
        _semantics.reset_cache()

    _reset_caches()
    yield
    _reset_caches()


def _ask(question: str) -> dict:
    return TestClient(app).post("/mi/query", json={"question": question}).json()


def _ledger(answer: dict) -> dict:
    return (answer.get("metadata") or {}).get("populationApplied") or {}


def _executed(answer: dict):
    return list(_ledger(answer).get("executed") or ())


def _facet(label: str, field_key: str) -> RequestedFacet:
    return RequestedFacet(kind=KIND_GEOGRAPHIC_SCOPE, label=label,
                          field_key=field_key)


# --------------------------------------------------------------------------- #
# 1-2. The question that was refused, and the one that always answered
# --------------------------------------------------------------------------- #
def test_a_temporal_geographic_filter_is_answered_and_proven():
    """The whole point: a trend narrowed to a place, with the place proven."""
    answer = _ask("Show funded balance over time for Scotland")
    assert answer["ok"] is True, answer.get("answer")
    assert "geographic scope was not applied" not in (answer.get("answer") or "")
    executed = _executed(answer)
    assert executed, "the route published no structural execution evidence"
    assert any(e["field"] == "collateral_geography"
               and [v.lower() for v in e["values"]] == ["scotland"]
               for e in executed), executed
    # The SAME population in every period, which is what makes the series one
    # series. `predicates_applied_in_every_period` publishes the intersection,
    # so a predicate that ran on only some frames never reaches this ledger.
    chart = next(a for a in answer["artifacts"] if a["type"] == "chart")
    assert len(chart["rows"]) == 3


def test_the_current_period_geographic_answer_is_unchanged():
    answer = _ask("Show funded balance for Scotland")
    assert answer["ok"] is True, answer.get("answer")
    assert "Scotland" in (answer.get("answer") or "")


# --------------------------------------------------------------------------- #
# 3-5. Falsification: the proof must REFUSE the wrong evidence
# --------------------------------------------------------------------------- #
def test_the_wrong_value_does_not_satisfy_a_place():
    """Asked Scotland, executed Wales, on the right field. MUST NOT pass.

    This is the property the repair exists for. Proving only that a geography
    field was filtered would stamp this APPLIED and publish a Welsh figure
    under a Scottish question.
    """
    ledger = {"executed": [{"field": "collateral_geography",
                            "canonical_field": "collateral_geography",
                            "op": "eq", "kind": "categorical",
                            "values": ["Wales"]}]}
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), ledger) is False


def test_the_right_value_on_the_wrong_basis_does_not_satisfy():
    """Asked BORROWER Scotland, executed COLLATERAL Scotland. MUST NOT pass.

    Both are "region" and they are different facts about a loan. The request
    resolved to the borrower field, so collateral evidence is not evidence for
    it — which is the whole reason the basis is resolved before execution.
    """
    ledger = {"executed": [{"field": "collateral_geography",
                            "canonical_field": "collateral_geography",
                            "op": "eq", "kind": "categorical",
                            "values": ["Scotland"]}]}
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "geographic_region_obligor"), ledger) is False


def test_no_synonym_is_invented_inside_the_receipt():
    """An ungoverned spelling is NOT equivalence, and this layer must not decide.

    Case is not a difference (the executor already compares case-insensitively).
    Anything beyond that belongs to the governed region owner that holds the
    ITL ladder; a synonym list grown here would be a second geography taxonomy,
    added one incident at a time and invisible to the ladder.
    """
    scots = {"executed": [{"field": "collateral_geography",
                           "canonical_field": "collateral_geography",
                           "op": "eq", "kind": "categorical",
                           "values": ["SCOTLAND"]}]}
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), scots) is True
    alba = {"executed": [{"field": "collateral_geography",
                          "canonical_field": "collateral_geography",
                          "op": "eq", "kind": "categorical",
                          "values": ["Alba"]}]}
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), alba) is False


def test_no_evidence_proves_nothing():
    """A route that publishes no structure leaves the facet exactly as it was."""
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), {}) is False
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), None) is False
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"),
        {"applied": ["collateral_geography (applied within each period)"]}) is False


# --------------------------------------------------------------------------- #
# The row count is context, never the test — in BOTH directions
# --------------------------------------------------------------------------- #
def test_a_predicate_that_moved_no_rows_was_still_applied():
    """A single-region book: every loan is in Scotland, so N -> N. Applied."""
    ledger = {"executed": [{"field": "collateral_geography",
                            "canonical_field": "collateral_geography",
                            "op": "eq", "kind": "categorical",
                            "values": ["Scotland"],
                            "rows_before": 640, "rows_after": 640}]}
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), ledger) is True


def test_a_predicate_that_left_no_rows_was_still_applied():
    """A book with nothing in Scotland: N -> 0. The place still ran."""
    ledger = {"executed": [{"field": "collateral_geography",
                            "canonical_field": "collateral_geography",
                            "op": "eq", "kind": "categorical",
                            "values": ["Scotland"],
                            "rows_before": 640, "rows_after": 0}]}
    assert receipt.geographic_scope_executed(
        _facet("Scotland", "collateral_geography"), ledger) is True


# --------------------------------------------------------------------------- #
# 6-10. Everything around it stays where it was
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question, field", [
    ("Show funded balance over time for the front book", "seasoning_segment"),
    ("Show funded balance over time for the back book", "seasoning_segment"),
])
def test_non_geographic_populations_are_unchanged(question, field):
    answer = _ask(question)
    assert answer["ok"] is True, answer.get("answer")
    assert any(e["field"] == field for e in _executed(answer)), _executed(answer)


def test_a_geographic_scope_and_a_threshold_are_both_proven():
    answer = _ask("Show balance over time for Scotland where LTV is over 50%")
    assert answer["ok"] is True, answer.get("answer")
    fields = {e["field"] for e in _executed(answer)}
    assert fields == {"collateral_geography", "current_loan_to_value"}, fields


def test_an_unknown_qualifier_still_fails_closed():
    answer = _ask("Show platinum balance over time for Scotland")
    assert answer["ok"] is False
    assert "platinum" in (answer.get("answer") or "").lower()


def test_the_evidence_is_the_intersection_across_periods():
    """A predicate that ran on only some frames is not evidence for the series.

    Written against the composer directly: a series whose points were measured
    over different populations must not publish one of them as though it were
    all of them.
    """
    from mi_agent_api import temporal_query

    scotland = {"field": "collateral_geography",
                "canonical_field": "collateral_geography", "op": "eq",
                "kind": "categorical", "values": ["Scotland"],
                "rows_before": 90, "rows_after": 22}
    wales = dict(scotland, values=["Wales"], rows_after=19)
    every = temporal_query.predicates_applied_in_every_period([
        {"appliedPredicates": [scotland]},
        {"appliedPredicates": [dict(scotland, rows_after=25)]},
    ])
    assert [e["values"] for e in every] == [["Scotland"]]
    assert "rows_after" not in every[0]
    mixed = temporal_query.predicates_applied_in_every_period([
        {"appliedPredicates": [scotland]},
        {"appliedPredicates": [wales]},
    ])
    assert mixed == []


def test_the_executor_records_what_it_compared():
    """The evidence begins at the execution owner, not at a re-parse."""
    from mi_agent.mi_query_executor import predicate_evidence

    class _Execution:
        normalised_value = "Scotland"
        applied_keys = ("collateral_geography",)
        kind = "categorical"
        resolved_op = "eq"

    entry = predicate_evidence(
        "collateral_geography", _Execution(),
        {"fields": {"collateral_geography":
                    {"canonical_field": "collateral_geography"}}}, 640, 52)
    assert entry["field"] == "collateral_geography"
    assert entry["op"] == "eq"
    assert entry["values"] == ["Scotland"]
    assert (entry["rows_before"], entry["rows_after"]) == (640, 52)


def test_an_explicit_borrower_basis_the_book_lacks_still_refuses(tmp_path,
                                                                 monkeypatch):
    """No silent substitution of collateral for a borrower basis that was named.

    The book below carries collateral geography and nothing else. "Borrower
    region" is not "region": the reader named a basis, and answering it from the
    property's location would be a different fact presented as the one asked
    for. The execution-evidence repair must not have opened that door — the
    facet resolves to the borrower field, so collateral evidence is not in the
    set that can satisfy it.
    """
    root = tmp_path / "collateral_only"
    for run_id, date, n in (("mi_2025_10", "2025-10-31", 90),
                            ("mi_2025_11", "2025-11-30", 95)):
        _write_run(root, run_id, date, n, 1.0)
        path = (root / "client_001" / run_id / "output" / "central"
                / "18_central_lender_tape.csv")
        frame = pd.read_csv(path)
        frame = frame.drop(columns=["geographic_region_obligor"])
        frame.to_csv(path, index=False)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    answer = _ask("Show balance by borrower region over time")
    assert answer["ok"] is False, answer.get("answer")


# --------------------------------------------------------------------------- #
# The time-axis owner claims its own wording, and claims nothing else
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("clause, claimed, unclaimed", [
    # The axis wording is claimed, so "over time" stops being read as a category
    # the book does not carry. The threshold beside it is untouched.
    ("balance by region over time for LTV over 50%", ("time", "over"), ("ltv",)),
    # No axis at all: nothing is claimed, and an age comparator stays readable.
    ("balance for borrowers over 85", (), ("over", "85", "borrowers")),
    # A WINDOW is not an axis wording, so the owner claims nothing here either.
    ("loan count by LTV bucket over the last three months where LTV is over 50%",
     (), ("months", "ltv")),
    # THE REGRESSION THAT KILLED THE FIRST ATTEMPT. A ranked movement question
    # names its axis with "month-on-month"; the first fix BLANKED that phrase
    # before the population reader saw the sentence, which left "added the most"
    # dangling and refused nine questions that had always answered. Scoped
    # ownership claims the axis words and leaves every other word exactly as
    # readable as it was.
    ("Which region added the most loans month-on-month?",
     ("month", "on"), ("added", "most", "loans", "region")),
])
def test_the_time_axis_owner_claims_only_its_own_wording(clause, claimed,
                                                         unclaimed):
    from mi_agent.llm_query_parser import _time_axis_words

    words = _time_axis_words(clause)
    for word in claimed:
        assert word in words, (word, words)
    for word in unclaimed:
        assert word not in words, (word, words)


def test_claiming_a_word_only_ever_suppresses_a_note():
    """An unclaimed word beside a claimed one still records its own note.

    The unknown-category test is `all()` over the captured value's words, so
    widening what the time-axis owner claims can never make an ungoverned
    qualifier disappear — which is the property that keeps "platinum" fail-closed
    while "time" stops being reported as a place the book does not carry.
    """
    from mi_agent.llm_query_parser import _claimed_by_an_owner

    axis = ("over", "time")
    assert _claimed_by_an_owner("time", {}, None, None, axis) is True
    assert _claimed_by_an_owner("platinum", {}, None, None, axis) is False
    # Unscoped, the same word is claimed by nobody.
    assert _claimed_by_an_owner("time", {}, None, None, ()) is False


def test_a_breakdown_over_time_sums_back_to_the_ungrouped_series():
    """Period by period, the parts are the whole.

    The strongest statement available about a period x dimension answer that
    does not require a second calculation: whatever the breakdown says, adding
    its categories up in each period must give the series the SAME question
    without a breakdown publishes. Both come from one executor over one frame,
    so a discrepancy would mean the grouping changed the population — which is
    the failure a temporal breakdown is most likely to hide, because each point
    looks plausible on its own.
    """
    from collections import defaultdict

    grouped = _ask("Show balance by region over time")
    plain = _ask("Show funded balance over time")
    assert grouped["ok"] is True and plain["ok"] is True

    table = next(a for a in grouped["artifacts"] if a["type"] == "table")
    composed = defaultdict(float)
    for row in table["rows"]:
        if row.get("value") is not None:
            composed[str(row["period"])] += float(row["value"])

    chart = next(a for a in plain["artifacts"] if a["type"] == "chart")
    shipped = {str(r["period"]): float(r["value"]) for r in chart["rows"]}

    assert set(composed) == set(shipped), set(composed) ^ set(shipped)
    for period, total in shipped.items():
        assert abs(composed[period] - total) < 0.011, (
            period, composed[period], total)


def test_the_breakdown_table_names_its_own_axis():
    """A reader must be able to tell WHICH axis was cut from the artifact alone.

    The column is the governed field key — the same one the current-period
    grouped table publishes — not a generic "category", so the temporal and the
    ordinary breakdown describe the same cut by the same name.
    """
    grouped = _ask("Show balance by region over time")
    table = next(a for a in grouped["artifacts"] if a["type"] == "table")
    keys = [c["key"] for c in table["columns"]]
    assert "collateral_geography" in keys, keys
    assert "category" not in keys, keys
