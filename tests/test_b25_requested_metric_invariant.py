#!/usr/bin/env python3
"""tests/test_b25_requested_metric_invariant.py — the B25 launch-gate fix.

The defect: "How does the direct book compare with the acquired book on borrower
age?" resolved the metric correctly, entered ``requested_metric`` mode, compared
NOTHING, and then answered

    "no governed directional differences were observed across the selected
     indicators"

with ok:true. An empty comparison set was rendered as a negative finding about
the two books.

Two invariants are pinned here:

1. **Requested-metric invariant.** If the caller names a metric, that metric must
   appear in the executed ``metric_comparisons`` before an answer may be given.
   Otherwise: a controlled refusal naming the metric and the reason.
2. **Empty-comparison safety.** ``metric_comparisons == []`` means NOTHING WAS
   COMPARED. It never means the portfolios are alike, and may never produce a
   negative conclusion.

Truth is computed here with pandas from the frames under test — the comparison
engine is never used as its own oracle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_workflows import portfolio_risk_comparison as prc
from mi_workflows.semantics import TAG_PORTFOLIO_COMPARISON, load_business_semantics

AGE = "youngest_borrower_age"
LTV = "current_loan_to_value"
BALANCE = "current_outstanding_balance"
PORTFOLIO = "portfolio_id"
DIRECT, ACQUIRED = "alp_origination", "alp_acquired"


class _Spec:
    """Minimal stand-in for the parsed spec the route hands the workflow."""

    def __init__(self, metric):
        self.metric = metric


@pytest.fixture(scope="module")
def bsr():
    return load_business_semantics()


@pytest.fixture(scope="module")
def book():
    """The real governed fixture — B25's own data, so the truth in this file is
    B25's truth and not a constructed analogue."""
    import os
    import logging

    from demo_platform import config as cfg

    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built — run "
                    "`python -m demo_platform.run_demo --generate --orchestrate`")
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    logging.disable(logging.WARNING)
    from mi_agent_api import data_source

    data_source.reset_cache()
    frame = data_source.get_dataframe()
    yield frame
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


#: The governed column that separates the two books on this tape.
COHORT = "source_portfolio_type"


def _with_asset_class(frame: pd.DataFrame) -> pd.DataFrame:
    """The same book, with the governed asset class the registry rule requires."""
    out = frame.copy()
    out[prc.ASSET_CLASS_FIELD] = "equity_release"
    return out


def _truth(frame: pd.DataFrame, column: str):
    """Mean per book, computed independently with pandas."""
    return (float(frame[frame[COHORT] == "direct"][column].mean()),
            float(frame[frame[COHORT] == "acquired"][column].mean()))


def _run(frame, question, metric, bsr):
    return prc.run_portfolio_risk_comparison(
        frame, question=question, bsr=bsr, mi_semantics={}, spec=_Spec(metric))


# =========================================================================== #
# Root cause, pinned so it cannot be misdiagnosed again
# =========================================================================== #
def test_borrower_age_is_governed_for_comparison_but_asset_specific(bsr):
    """The rule that used to block B25, still intact.

    Borrower age is equity-release-specific. It is compared only when the books
    declare that asset class — which they now do, because onboarding publishes
    it to the governed portfolio registry (see
    ``tests/test_asset_class_lifecycle.py``). The rule was never the defect and
    is unchanged; what changed is that the asset class now reaches it."""
    entry = bsr.get(AGE)
    assert entry is not None
    assert entry.tagged(TAG_PORTFOLIO_COMPARISON)
    assert entry.portfolio_comparability == "comparable"
    assert entry.asset_applicability == ("equity_release",)
    assert not entry.is_cross_asset()

    columns = (AGE, LTV, BALANCE, PORTFOLIO)
    declined = prc._comparability_decision(entry, None, columns)
    assert declined.selected is False
    assert "asset-specific" in declined.reason

    # ...and with a governed shared asset class the SAME field is selected. The
    # capability exists; only the governed evidence for it is absent.
    assert prc._comparability_decision(entry, "equity_release", columns).selected


def test_the_rule_that_blocks_age_protects_many_other_fields(bsr):
    """Relaxing it to make one question pass would let SME and CRE measures be
    compared on any book. That is why the fix is an invariant, not a bypass."""
    asset_specific = {e.source_field for e in bsr.for_workflow(TAG_PORTFOLIO_COMPARISON)
                      if not e.is_cross_asset()}
    assert len(asset_specific) > 20
    for field in ("enterprise_value", "earnings_before_interest_taxes_ebit",
                  "negative_equity_guarantee"):
        assert field in asset_specific


# =========================================================================== #
# 1. The requested-metric invariant
# =========================================================================== #
#: An SME-only governed measure. On an equity-release book it is correctly not
#: comparable, which makes it the honest way to exercise the invariant now that
#: borrower age IS comparable here.
OTHER_ASSET_METRIC = "enterprise_value"


def test_a_requested_metric_that_is_not_compared_is_reported_as_such(bsr, book):
    """The outcome of a requested metric is always stated, never dropped
    silently into an empty comparison list.

    Exercised with an SME measure on an equity-release book: the governed rule
    declines it, and that decline must reach the caller."""
    result = _run(book, "How do the two books compare on enterprise value?",
                  OTHER_ASSET_METRIC, bsr)
    assert result["available"] is True
    requested = result["requested_metric"]
    assert requested["field"] == OTHER_ASSET_METRIC
    assert requested["compared"] is False
    assert result["metric_comparisons"] == []
    assert any(OTHER_ASSET_METRIC in note and "not compared" in note
               for note in result["limitations"])


def test_with_a_governed_asset_class_the_age_comparison_executes_and_reconciles(bsr, book):
    """The capability is real: given the governed evidence the rule requires,
    the requested age comparison runs and matches pandas exactly.

    This is the evidence behind the outstanding decision recorded in the B25
    report — the question needs a governed asset class for the client's
    portfolios, not a code change."""
    frame = _with_asset_class(book)
    result = _run(frame, "How does the direct book compare with the acquired "
                         "book on borrower age?", AGE, bsr)
    assert result["requested_metric"]["compared"] is True
    comparisons = result["metric_comparisons"]
    assert [c["field"] for c in comparisons] == [AGE]

    expected_direct, expected_acquired = _truth(frame, AGE)
    got_direct = comparisons[0]["portfolio_a"]["value"]
    got_acquired = comparisons[0]["portfolio_b"]["value"]
    assert got_direct == pytest.approx(expected_direct, rel=1e-9)
    assert got_acquired == pytest.approx(expected_acquired, rel=1e-9)
    assert got_direct - got_acquired == pytest.approx(
        expected_direct - expected_acquired, rel=1e-9)


# =========================================================================== #
# 4. A different governed metric — the fix is not wired to age
# =========================================================================== #
def test_a_cross_asset_metric_still_compares_and_reconciles(bsr, book):
    """LTV is cross-asset, so it compares on the same frame with no asset class
    declared. The invariant is generic, not age-specific."""
    result = _run(book, "What is the average LTV of the direct book vs the "
                        "acquired book?", LTV, bsr)
    assert result["requested_metric"] == {
        "field": LTV, "display_name": "Current Loan To Value",
        "compared": True, "reason": "governed for portfolio comparison"}
    assert [c["field"] for c in result["metric_comparisons"]] == [LTV]


# =========================================================================== #
# 5. An unsupported requested metric is named, never substituted
# =========================================================================== #
def test_an_ungoverned_requested_metric_is_named_and_not_substituted(bsr, book):
    result = _run(book, "Compare the books on something ungoverned",
                  "not_a_governed_field", bsr)
    requested = result["requested_metric"]
    assert requested["field"] == "not_a_governed_field"
    assert requested["compared"] is False
    assert result["metric_comparisons"] == []
    assert any("not_a_governed_field" in note for note in result["limitations"])


# =========================================================================== #
# End to end through the governed capability
# =========================================================================== #
@pytest.fixture(scope="module")
def ask(book):
    from trakt_core.context import ExecutionContext
    from demo_platform import config as cfg
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache = {}

    def _ask(question):
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


#: Wordings that must all reach the same governed outcome. The first is B25.
AGE_PHRASINGS = [
    "How does the direct book compare with the acquired book on borrower age?",
    "Compare borrower age between the direct and acquired books.",
    "What is the average borrower age in Direct versus Acquired?",
]

#: Phrases a negative conclusion would use. None may appear when nothing was
#: compared — this is the exact sentence the defect produced.
NEGATIVE_CONCLUSIONS = (
    "no governed directional differences were observed",
    "no directional differences",
    "no differences observed",
    "broadly similar",
    "the portfolios are the same",
)


@pytest.mark.parametrize("question", AGE_PHRASINGS, ids=[q[:46] for q in AGE_PHRASINGS])
def test_an_uncompared_metric_never_produces_a_negative_conclusion(ask, question):
    """THE B25 CONTRACT. Borrower age is not compared on this book, so the
    answer must say so — and must not assert that the books do not differ."""
    envelope = ask(question)
    answer = (envelope.get("answer") or envelope.get("error") or "").lower()
    assert answer
    for phrase in NEGATIVE_CONCLUSIONS:
        assert phrase not in answer, f"negative conclusion leaked: {answer}"


def test_b25_now_compares_borrower_age(ask, book):
    """B25, answered. The truth reconciliation lives in
    ``tests/test_asset_class_lifecycle.py``; this pins the declared evidence."""
    envelope = ask(AGE_PHRASINGS[0])
    assert envelope.get("ok") is True, envelope.get("error")
    declared = (envelope.get("metadata") or {}).get("portfolioComparison") or {}
    assert declared.get("requestedMetric") == AGE
    assert declared.get("requestedMetricCompared") is True
    assert declared.get("measuresCompared") == [AGE]
    assert "Youngest Borrower Age" in (
        (envelope.get("executionSummary") or {}).get("receipt") or "")


def test_a_metric_outside_the_books_asset_class_is_refused_and_named(book):
    """The refusal path, still live.

    Built by declaring a DIFFERENT governed asset class for the books, which is
    the realistic shape: an equity-release measure asked of an SME book. The
    route must name the measure, substitute nothing, and draw no negative
    conclusion.
    """
    import logging
    import os

    import yaml
    from trakt_core.context import ExecutionContext
    from demo_platform import config as cfg

    registry_file = Path(os.environ["MI_AGENT_SCRATCH"]) / "sme_registry.yaml"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(yaml.safe_dump({"portfolios": [
        {"source_portfolio_id": "alp_origination", "asset_class": "sme"},
        {"source_portfolio_id": "alp_acquired", "asset_class": "sme"}]}))

    before = dict(os.environ)
    os.environ["TRAKT_PORTFOLIO_REGISTRY"] = str(registry_file)
    logging.disable(logging.WARNING)
    try:
        from mi_agent_api import data_source
        from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

        data_source.reset_cache()
        envelope = execute_governed_mi_query(
            MiQueryRequest(question=AGE_PHRASINGS[0]),
            ExecutionContext.for_internal(cfg.CLIENT_ID)).result
    finally:
        logging.disable(logging.NOTSET)
        os.environ.clear()
        os.environ.update(before)
        from mi_agent_api import data_source as _ds
        _ds.reset_cache()

    assert envelope.get("ok") is False
    answer = envelope["answer"].lower()
    assert "borrower age" in answer
    assert not (envelope.get("artifacts") or [])
    for phrase in NEGATIVE_CONCLUSIONS:
        assert phrase not in answer
    # …and no other measure was offered in its place.
    assert "loan to value" not in answer and "ltv" not in answer
    declared = (envelope.get("metadata") or {}).get("portfolioComparison") or {}
    assert declared.get("requestedMetricCompared") is False
    assert declared.get("measuresCompared") == []


def test_a_supported_comparison_still_answers_and_names_its_measure(ask, book):
    """Proof the fix is an invariant and not a blanket refusal: the cross-asset
    LTV comparison answers, and its receipt states what was compared."""
    envelope = ask("What is the average LTV of the direct book vs the acquired book?")
    assert envelope.get("ok") is True, envelope.get("error")
    receipt = (envelope.get("executionSummary") or {}).get("receipt") or ""
    assert "Current Loan To Value" in receipt
    assert "Direct vs Acquired" in receipt
    assert receipt != "Calculated: Portfolio comparison."

    comparisons = envelope["workflow"]["metric_comparisons"]
    expected_direct, expected_acquired = _truth(book, LTV)
    # The engine weights LTV by balance; assert against its declared basis
    # rather than an unweighted mean.
    assert comparisons[0]["aggregation"] == "weighted_average"
    frame = book.dropna(subset=[LTV, BALANCE])
    for key, cohort in (("portfolio_a", "direct"), ("portfolio_b", "acquired")):
        side = frame[frame[COHORT] == cohort]
        weighted = float((side[LTV] * side[BALANCE]).sum() / side[BALANCE].sum())
        assert comparisons[0][key]["value"] == pytest.approx(weighted, rel=1e-9)
    assert expected_direct > 0 and expected_acquired > 0   # sanity on the fixture


def test_the_p0_guard_marks_the_cohort_facet_lost_when_the_measure_was_dropped():
    """The guard must distinguish "the comparison route ran" from "the measure
    the question named was compared". Constructed directly, so the check is
    tested independently of the route being right today."""
    from mi_agent import execution_receipt as rc

    envelope = {"metadata": {"portfolioComparison": {
        "requestedMetric": AGE, "requestedMetricCompared": False,
        "measuresCompared": [], "portfolioA": "Direct", "portfolioB": "Acquired"}}}
    facets = rc.reconcile_routed_facets(
        [rc.RequestedFacet(kind=rc.KIND_COHORT_COMPARISON,
                           label="a comparison between two books")],
        route="portfolio_risk_comparison", semantics={"fields": {}},
        envelope=envelope)
    assert facets[0].status == rc.LOST
    assert "named" in facets[0].reason


def test_the_p0_guard_marks_the_cohort_facet_lost_when_nothing_was_compared():
    from mi_agent import execution_receipt as rc

    envelope = {"metadata": {"portfolioComparison": {
        "requestedMetric": None, "requestedMetricCompared": None,
        "measuresCompared": [], "dimensionsCompared": [],
        "portfolioA": "Direct", "portfolioB": "Acquired"}}}
    facets = rc.reconcile_routed_facets(
        [rc.RequestedFacet(kind=rc.KIND_COHORT_COMPARISON,
                           label="a comparison between two books")],
        route="portfolio_risk_comparison", semantics={"fields": {}},
        envelope=envelope)
    assert facets[0].status == rc.LOST
    assert "nothing was measured" in facets[0].reason


def test_the_p0_guard_accepts_a_comparison_that_measured_something():
    from mi_agent import execution_receipt as rc

    envelope = {"metadata": {"portfolioComparison": {
        "requestedMetric": LTV, "requestedMetricCompared": True,
        "measuresCompared": [LTV], "measureLabels": ["Current Loan To Value"],
        "aggregations": ["weighted_average"],
        "portfolioA": "Direct", "portfolioB": "Acquired"}}}
    facets = rc.reconcile_routed_facets(
        [rc.RequestedFacet(kind=rc.KIND_COHORT_COMPARISON,
                           label="a comparison between two books")],
        route="portfolio_risk_comparison", semantics={"fields": {}},
        envelope=envelope)
    assert facets[0].status == rc.APPLIED
    receipt = rc.build_routed_receipt(route="portfolio_risk_comparison",
                                      envelope=envelope, facets=facets)
    assert "Weighted-average Current Loan To Value" in receipt.render()
    assert "Direct vs Acquired" in receipt.render()
