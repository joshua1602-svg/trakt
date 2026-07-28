"""Portfolio Risk Comparison — the pure governed workflow.

Covers the workflow package (``mi_workflows``): recognition predicates, scope
resolution, common-reporting-date enforcement, Business Semantics Registry
consumption (comparability, asset applicability, aggregation, weights, share
bases, directionality), the shared calculation engine (weighted averages,
shares, distributions, the mixed-currency guard), evidence, audit, and the
observational-summary discipline. The API adapter and recogniser registration
are covered separately in
``mi_agent_api/tests/test_portfolio_risk_comparison_route.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_workflows import engine
from mi_workflows import portfolio_risk_comparison as prc
from mi_workflows.semantics import (
    load_business_semantics,
    semantics_context_resolver,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bsr():
    return load_business_semantics(REPO / "config" / "business_semantics_registry.yaml")


#: Runtime MI-semantics stub carrying only the unit formats the tests need —
#: the same shape ``load_mi_semantics`` produces.
MI_SEMANTICS = {"fields": {
    "current_outstanding_balance": {"format": "currency"},
    "arrears_balance": {"format": "currency"},
    "collateral_value": {"format": "currency"},
    "current_loan_to_value": {"format": "percent"},
    "current_interest_rate": {"format": "percent"},
    "borrower_1_age": {"format": "integer"},
}}


def _frame(**overrides) -> pd.DataFrame:
    """Two governed portfolios, four loans each, one reporting date."""
    base = {
        "source_portfolio_id": ["direct_001"] * 4 + ["acquired_001"] * 4,
        "source_portfolio_type": ["direct"] * 4 + ["acquired"] * 4,
        "reporting_date": ["2026-06-30"] * 8,
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0,
                                        150.0, 250.0, 350.0, 450.0],
        "current_loan_to_value": [0.30, 0.40, 0.50, 0.60,
                                  0.55, 0.65, 0.70, 0.75],
        "arrears_balance": [0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 20.0, 15.0],
        "collateral_value": [500.0, 600.0, 700.0, 800.0,
                             400.0, 500.0, 600.0, 700.0],
        "current_interest_rate": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
        "exposure_currency_denomination": ["GBP"] * 8,
        "account_status": ["performing", "performing", "arrears", "performing",
                           "performing", "arrears", "arrears", None],
        "product_type": ["lump", "lump", "draw", "draw",
                         "lump", "draw", "draw", "draw"],
        "credit_impaired_obligor": ["N", "N", "Y", "N", "Y", "Y", "N", None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _run(df, question="Compare direct_001 with acquired_001", *, bsr, **kw):
    kw.setdefault("mi_semantics", MI_SEMANTICS)
    kw.setdefault("as_of", "2026-06-30")
    return prc.run_portfolio_risk_comparison(df, question=question, bsr=bsr, **kw)


def _metric(result, field):
    for c in result["metric_comparisons"]:
        if c["field"] == field:
            return c
    return None


class _Spec:
    def __init__(self, metric=None, temporal_mode=None, forecast_mode=None):
        self.metric = metric
        self.temporal_mode = temporal_mode
        self.forecast_mode = forecast_mode


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
class TestRecognition:
    @pytest.mark.parametrize("question", [
        "Compare SPV I with SPV II",
        "Compare the originated and acquired books",
        "Which portfolio has higher LTV?",
        "Compare credit quality across portfolios",
        "Compare payment performance between the portfolios",
        "Show differences between warehouse pools",
        "Compare geographic composition of the books",
        "Compare product mix across the SPVs",
        "direct_001 versus acquired_001",
        "Which portfolio appears stronger across the selected indicators?",
    ])
    def test_portfolio_comparison_language_is_recognised(self, question):
        matched, reason = prc.is_portfolio_comparison_question(question)
        assert matched, reason

    @pytest.mark.parametrize("question", [
        # period change / temporal — owned by other routes
        "Compare October and November funded balance",
        "What has changed versus the prior month?",
        "Compare the book to last year",
        # concentration
        "Compare concentration across portfolios",
        "Which portfolio has the largest exposures?",
        # forecast
        "Forecast which portfolio will be bigger",
        # covenant
        "Compare covenant headroom across portfolios",
        "Is the portfolio within appetite limits?",
        # eligibility
        "Which loans are eligible for the warehouse?",
        # raw exports / reconciliation
        "Export the loan level data for both portfolios",
        "Reconcile the transactions between the books",
        # not a comparison at all
        "What is the average LTV?",
        "Summarise the portfolio",
    ])
    def test_questions_owned_elsewhere_are_rejected(self, question):
        matched, reason = prc.is_portfolio_comparison_question(question)
        assert not matched, f"stole {question!r} ({reason})"

    def test_a_parsed_cross_period_intent_is_rejected_even_with_portfolio_words(self):
        matched, reason = prc.is_portfolio_comparison_question(
            "Compare the portfolios", _Spec(temporal_mode="compare"))
        assert not matched
        assert "cross-period" in reason

    def test_a_parsed_forecast_intent_is_rejected(self):
        matched, _ = prc.is_portfolio_comparison_question(
            "Compare the portfolios", _Spec(forecast_mode="extrapolation"))
        assert not matched


# --------------------------------------------------------------------------- #
# Scope resolution
# --------------------------------------------------------------------------- #
class TestScopeResolution:
    def test_two_named_cohorts_resolve_to_two_scopes(self, bsr):
        result = _run(_frame(), "Compare direct_001 with acquired_001", bsr=bsr)
        assert result["available"] is True
        sides = result["portfolio_results"]
        assert [s["context_id"] for s in sides] == ["direct_001", "acquired_001"]
        assert [s["row_count"] for s in sides] == [4, 4]

    def test_originated_vs_acquired_resolves_type_groups(self, bsr):
        result = _run(_frame(), "Compare the originated and acquired books", bsr=bsr)
        scope = result["comparison_scope"]
        assert scope["portfolio_a"]["context_id"] == "direct"
        assert scope["portfolio_b"]["context_id"] == "acquired"

    def test_generic_comparison_uses_the_governed_type_groups(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        scope = result["comparison_scope"]
        assert scope["portfolio_a"]["context_id"] == "direct"
        assert scope["portfolio_b"]["context_id"] == "acquired"

    def test_unknown_portfolio_is_a_controlled_failure_listing_governed_ids(self, bsr):
        result = _run(_frame(), "Compare direct_001 with acquired_009", bsr=bsr)
        assert result["available"] is False
        assert "acquired_009" in result["reason"]
        assert "direct_001" in result["reason"]          # what IS available

    def test_more_than_two_scopes_is_a_controlled_failure(self, bsr):
        result = _run(_frame(),
                      "Compare direct_001, acquired_001 and acquired_002", bsr=bsr)
        assert result["available"] is False
        assert "pairwise" in result["reason"]

    def test_single_portfolio_deployment_is_a_controlled_failure(self, bsr):
        df = _frame(source_portfolio_id=["direct_001"] * 8,
                    source_portfolio_type=["direct"] * 8)
        result = _run(df, "Compare the portfolios", bsr=bsr)
        assert result["available"] is False
        assert "nothing to compare" in result["reason"]

    def test_no_provenance_is_a_controlled_failure(self, bsr):
        df = _frame()
        df = df.drop(columns=["source_portfolio_id", "source_portfolio_type"])
        result = _run(df, "Compare the portfolios", bsr=bsr)
        assert result["available"] is False
        assert "provenance" in result["reason"]

    def test_empty_scope_is_a_controlled_failure(self, bsr):
        # The registry knows both portfolios (e.g. via governed metadata), but
        # the acquired book has no rows on the active dataset.
        from mi_agent.portfolio_scope import registry_for_frame

        registry = registry_for_frame(_frame(), client_id="c")
        df = _frame()
        df = df[df["source_portfolio_id"] == "direct_001"]  # acquired empty
        result = _run(df, "Compare direct_001 with acquired_001", bsr=bsr,
                      registry=registry)
        assert result["available"] is False
        assert "no rows in scope" in result["reason"]


# --------------------------------------------------------------------------- #
# Common reporting date
# --------------------------------------------------------------------------- #
class TestReportingDate:
    def test_common_date_is_reported(self, bsr):
        result = _run(_frame(), bsr=bsr)
        assert result["reporting_date"] == "2026-06-30"
        assert result["audit"]["reporting_date"] == "2026-06-30"

    def test_different_reporting_dates_refuse_comparison(self, bsr):
        df = _frame()
        df.loc[df.source_portfolio_id == "acquired_001",
               "reporting_date"] = "2026-05-31"
        result = _run(df, bsr=bsr)
        assert result["available"] is False
        assert "different reporting dates" in result["reason"]

    def test_a_scope_spanning_dates_refuses_comparison(self, bsr):
        df = _frame()
        df.loc[df.index[:2], "reporting_date"] = "2026-05-31"
        result = _run(df, bsr=bsr)
        assert result["available"] is False
        assert "multiple reporting dates" in result["reason"]

    def test_missing_reporting_date_column_is_disclosed_not_fatal(self, bsr):
        df = _frame().drop(columns=["reporting_date"])
        result = _run(df, bsr=bsr, as_of="2026-06-30")
        assert result["available"] is True
        assert result["reporting_date"] == "2026-06-30"
        assert any("no reporting_date column" in w for w in result["warnings"])


# --------------------------------------------------------------------------- #
# Comparability + asset applicability (the registry decides)
# --------------------------------------------------------------------------- #
class TestComparability:
    def test_scale_aligned_fields_are_excluded_with_an_explanation(self, bsr):
        # erm_product_type is an originator vocabulary the registry marks
        # requires_scale_alignment; it must be excluded with an explanation,
        # never compared through a heuristic mapping.
        assert (bsr.get("erm_product_type").portfolio_comparability
                == "requires_scale_alignment")
        df = _frame(erm_product_type=["Lifetime"] * 8)
        result = _run(df, "Compare product mix across the portfolios", bsr=bsr)
        excluded = {e["field"]: e for e in result["audit"]["fields_excluded"]}
        assert "erm_product_type" in excluded
        assert "scale" in excluded["erm_product_type"]["reason"]
        assert all(c["field"] != "erm_product_type"
                   for c in result["distribution_comparisons"])

    def test_not_comparable_fields_are_never_compared(self, bsr):
        df = _frame(pd_previous=[0.01] * 8)
        result = _run(df, "Compare the portfolios", bsr=bsr)
        assert _metric(result, "pd_previous") is None

    def test_within_asset_class_only_needs_a_shared_declared_class(self, bsr):
        entry = bsr.get("contractual_annual_rental_income")
        assert entry.portfolio_comparability == "within_asset_class_only"
        df = _frame(contractual_annual_rental_income=[10.0] * 8)
        result = _run(df, "Compare rental income across the portfolios",
                      bsr=bsr, spec=_Spec(metric="contractual_annual_rental_income"))
        decision = {e["field"]: e for e in result["audit"]["comparability_decisions"]}
        assert decision["contractual_annual_rental_income"]["selected"] is False
        assert "asset class" in decision["contractual_annual_rental_income"]["reason"]

        with_class = _frame(contractual_annual_rental_income=[10.0] * 8,
                            asset_class=["commercial_real_estate"] * 8)
        result = _run(with_class, "Compare rental income across the portfolios",
                      bsr=bsr, spec=_Spec(metric="contractual_annual_rental_income"))
        assert _metric(result, "contractual_annual_rental_income") is not None

    def test_asset_specific_metric_needs_matching_asset_classes(self, bsr):
        entry = bsr.get("borrower_1_age")
        assert entry.asset_applicability == ("equity_release",)
        shared = _frame(borrower_1_age=[70, 71, 72, 73, 74, 75, 76, 77],
                        asset_class=["equity_release"] * 8)
        result = _run(shared, "Compare the portfolios", bsr=bsr,
                      spec=_Spec(metric="borrower_1_age"))
        assert _metric(result, "borrower_1_age") is not None

        mismatched = _frame(borrower_1_age=[70, 71, 72, 73, 74, 75, 76, 77],
                            asset_class=["equity_release"] * 4 + ["sme"] * 4)
        result = _run(mismatched, "Compare the portfolios", bsr=bsr,
                      spec=_Spec(metric="borrower_1_age"))
        assert _metric(result, "borrower_1_age") is None
        assert any("different asset classes" in note
                   for note in result["limitations"])

    def test_cross_asset_metrics_survive_an_asset_mismatch(self, bsr):
        df = _frame(asset_class=["equity_release"] * 4 + ["sme"] * 4)
        result = _run(df, "Compare the portfolios", bsr=bsr)
        assert result["available"] is True
        assert _metric(result, "current_loan_to_value") is not None

    def test_ungoverned_metric_is_explained_not_compared(self, bsr):
        df = _frame(some_bespoke_score=[1] * 8)
        result = _run(df, "Compare the portfolios", bsr=bsr,
                      spec=_Spec(metric="some_bespoke_score"))
        assert result["metric_comparisons"] == []
        assert any("not governed" in note for note in result["limitations"])


# --------------------------------------------------------------------------- #
# Currency guard
# --------------------------------------------------------------------------- #
class TestCurrencyGuard:
    def test_currency_mismatch_suppresses_monetary_and_keeps_the_rest(self, bsr):
        df = _frame()
        df.loc[df.source_portfolio_id == "acquired_001",
               "exposure_currency_denomination"] = "EUR"
        result = _run(df, "Compare the portfolios", bsr=bsr)
        assert result["available"] is True
        monetary = [c["field"] for c in result["metric_comparisons"]
                    if c["unit"] == "currency"]
        assert monetary == []
        assert _metric(result, "current_loan_to_value") is not None
        assert any("no FX conversion" in note for note in result["limitations"])
        suppressed = [e for e in result["audit"]["fields_excluded"]
                      if "currency guard" in e["reason"]]
        assert {e["field"] for e in suppressed} >= {"current_outstanding_balance",
                                                    "arrears_balance"}

    def test_internally_mixed_scope_suppresses_monetary(self, bsr):
        df = _frame()
        df.loc[df.index[:2], "exposure_currency_denomination"] = "USD"
        result = _run(df, "Compare the portfolios", bsr=bsr)
        assert all(c["unit"] != "currency" for c in result["metric_comparisons"])
        assert any("mixed currencies within" in note
                   for note in result["limitations"])

    def test_matching_currencies_allow_monetary_comparison(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        balance = _metric(result, "current_outstanding_balance")
        assert balance is not None
        assert balance["portfolio_a"]["value"] == pytest.approx(1000.0)
        assert balance["portfolio_b"]["value"] == pytest.approx(1200.0)
        assert result["audit"]["currencies"]["monetary_comparison_allowed"] is True


# --------------------------------------------------------------------------- #
# Calculations through the shared engine
# --------------------------------------------------------------------------- #
class TestCalculations:
    def test_weighted_average_uses_the_registry_weight_field(self, bsr):
        result = _run(_frame(), bsr=bsr)
        ltv = _metric(result, "current_loan_to_value")
        assert ltv["aggregation"] == "weighted_average"
        assert ltv["weight_basis"] == "current_outstanding_balance"
        # Hand-computed: Σ(ltv·bal)/Σbal for direct_001.
        expected_a = (0.30 * 100 + 0.40 * 200 + 0.50 * 300 + 0.60 * 400) / 1000.0
        assert ltv["portfolio_a"]["value"] == pytest.approx(expected_a)
        assert ltv["portfolio_a"]["denominator"] == pytest.approx(1000.0)
        assert ltv["difference"]["absolute"] == pytest.approx(
            ltv["portfolio_a"]["value"] - ltv["portfolio_b"]["value"])

    def test_weighted_average_excludes_null_and_nonpositive_weights(self):
        df = pd.DataFrame({"m": [0.5, 0.7, 0.9, 0.2],
                           "w": [100.0, None, 0.0, 300.0]})
        agg = engine.aggregate(df, "m", "weighted_average", weight_field="w")
        assert agg.valid_population == 2
        assert agg.excluded_population == 2
        assert agg.value == pytest.approx((0.5 * 100 + 0.2 * 300) / 400.0)
        assert agg.denominator == pytest.approx(400.0)

    def test_share_metric_uses_the_registry_share_basis(self, bsr):
        result = _run(_frame(), "Compare credit quality across the portfolios",
                      bsr=bsr)
        share = _metric(result, "credit_impaired_obligor")
        assert share is not None
        assert share["aggregation"] == "share"
        assert share["weight_basis"] == "count"
        assert share["portfolio_a"]["value"] == pytest.approx(1 / 4)
        # One null flag in acquired_001: excluded from the basis, disclosed.
        assert share["portfolio_b"]["value"] == pytest.approx(2 / 3)
        assert share["portfolio_b"]["excluded_population"] == 1

    def test_directionality_interpretations_follow_the_registry(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        arrears = _metric(result, "arrears_balance")          # higher_is_worse
        assert arrears["directionality"]["declared"] == "higher_is_worse"
        assert arrears["directionality"]["higher"] == "portfolio_b"
        assert (arrears["directionality"]["interpretation"]
                == engine.FAVOURS_A)
        collateral = _metric(result, "collateral_value")      # higher_is_better
        assert (collateral["directionality"]["interpretation"]
                == engine.FAVOURS_A)
        balance = _metric(result, "current_outstanding_balance")  # neutral
        assert balance["directionality"]["interpretation"] == engine.NEUTRAL
        rate = _metric(result, "current_interest_rate")       # context_dependent
        assert (rate["directionality"]["interpretation"]
                == engine.CONTEXT_DEPENDENT)

    def test_equal_values_are_no_observed_difference(self):
        verdict = engine.directionality_verdict("higher_is_worse", 1.0, 1.0)
        assert verdict["interpretation"] == engine.NO_DIFFERENCE
        assert verdict["higher"] is None

    def test_no_comparison_formula_is_invented(self):
        assert engine.compare_values(None, 1.0) == {"absolute": None,
                                                    "relative": None}
        assert engine.compare_values(3.0, 2.0)["absolute"] == pytest.approx(1.0)
        assert engine.compare_values(3.0, 2.0)["relative"] == pytest.approx(0.5)
        assert engine.compare_values(3.0, 0.0)["relative"] is None


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
class TestDistributions:
    def test_count_exposure_and_unknown_shares(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        status = next(d for d in result["distribution_comparisons"]
                      if d["field"] == "account_status")
        # acquired_001 has one null status → explicit unknown share, not a drop.
        assert status["unknown_share"]["portfolio_a"] == pytest.approx(0.0)
        assert status["unknown_share"]["portfolio_b"] == pytest.approx(1 / 4)
        rows = {r["category"]: r for r in status["categories"]}
        assert rows["performing"]["count_share_a"] == pytest.approx(3 / 4)
        assert rows["performing"]["count_share_b"] == pytest.approx(1 / 4)
        assert engine.UNKNOWN_CATEGORY in rows
        # Exposure shares are within-portfolio ratios over the governed weight.
        assert rows["arrears"]["exposure_share_a"] == pytest.approx(300 / 1000)

    def test_categories_missing_from_one_side_are_explicit(self, bsr):
        df = _frame(product_type=["lump"] * 4 + ["draw"] * 4)
        result = _run(df, "Compare product mix across the portfolios", bsr=bsr)
        product = next(d for d in result["distribution_comparisons"]
                       if d["field"] == "product_type")
        rows = {r["category"]: r for r in product["categories"]}
        assert rows["lump"]["present_in"] == ["portfolio_a"]
        assert rows["lump"]["count_share_b"] == 0.0
        assert rows["draw"]["present_in"] == ["portfolio_b"]

    def test_different_portfolio_sizes_compare_on_shares(self, bsr):
        df = pd.concat([_frame(), _frame().iloc[4:]], ignore_index=True)
        result = _run(df, "Compare the portfolios", bsr=bsr)
        status = next(d for d in result["distribution_comparisons"]
                      if d["field"] == "account_status")
        assert status["population"]["portfolio_a"] == 4
        assert status["population"]["portfolio_b"] == 8
        for row in status["categories"]:
            for key in ("count_share_a", "count_share_b"):
                assert row[key] is None or 0.0 <= row[key] <= 1.0

    def test_no_concentration_metrics_are_produced(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        text = json.dumps(result).lower()
        for forbidden in ("herfindahl", "hhi", "top_n", "concentration_index"):
            assert forbidden not in text


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
class TestModes:
    def test_requested_metric_mode(self, bsr):
        result = _run(_frame(), "Which portfolio has higher LTV?", bsr=bsr,
                      spec=_Spec(metric="current_loan_to_value"))
        assert result["comparison_scope"]["mode"] == "requested_metric"
        assert [c["field"] for c in result["metric_comparisons"]] == [
            "current_loan_to_value"]

    def test_requested_concept_mode(self, bsr):
        result = _run(_frame(), "Compare payment performance across the portfolios",
                      bsr=bsr)
        assert result["comparison_scope"]["mode"] == "requested_concept"
        assert all(c["concept"] == "payment_performance"
                   for c in result["metric_comparisons"])
        assert _metric(result, "arrears_balance") is not None

    def test_overview_mode_is_bounded_and_deterministic(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        assert result["comparison_scope"]["mode"] == "overview"
        by_concept: dict = {}
        for c in result["metric_comparisons"]:
            by_concept.setdefault(c["concept"], []).append(c["field"])
        for concept, fields in by_concept.items():
            assert len(fields) <= prc.OVERVIEW_MEASURES_PER_CONCEPT, concept

    def test_deterministic_output(self, bsr):
        one = _run(_frame(), "Compare the portfolios", bsr=bsr)
        two = _run(_frame(), "Compare the portfolios", bsr=bsr)
        assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


# --------------------------------------------------------------------------- #
# Summary discipline, evidence, audit
# --------------------------------------------------------------------------- #
class TestSummaryEvidenceAudit:
    def test_summary_is_observational_never_a_judgement(self, bsr):
        result = _run(_frame(), "Which portfolio appears stronger?", bsr=bsr)
        text = " ".join(result["summary"]).lower()
        for forbidden in ("safer", "higher quality", "preferable", "breach",
                          "appetite", "compliant", "acceptable", "risk score"):
            assert forbidden not in text
        assert "not an overall assessment" in " ".join(result["summary"])

    def test_no_composite_score_in_the_contract(self, bsr):
        result = _run(_frame(), "Compare the portfolios", bsr=bsr)
        text = json.dumps(result).lower()
        assert "composite" not in text
        assert "overall_score" not in text

    def test_evidence_discloses_populations_and_denominators(self, bsr):
        result = _run(_frame(), bsr=bsr)
        ltv_evidence = next(e for e in result["evidence"]
                            if e["field"] == "current_loan_to_value")
        assert ltv_evidence["kind"] == "metric_population"
        assert ltv_evidence["portfolio_a"]["population"] == 4
        assert ltv_evidence["portfolio_a"]["valid_population"] == 4
        assert ltv_evidence["portfolio_a"]["denominator"] == pytest.approx(1000.0)
        assert ltv_evidence["source"] == "funded canonical tape"

    def test_audit_records_the_governed_decision_trail(self, bsr):
        result = _run(_frame(), bsr=bsr)
        audit = result["audit"]
        assert audit["workflow_id"] == prc.WORKFLOW_ID
        assert audit["bsr_version"] == bsr.version
        assert audit["bsr_schema_version"] == bsr.schema_version
        assert audit["calculation_version"] == engine.CALCULATION_VERSION
        assert audit["dataset"] == "funded"
        assert audit["reporting_date"] == "2026-06-30"
        assert audit["selected_portfolios"]["portfolio_a"]["context_id"] == "direct_001"
        assert audit["currencies"]["portfolio_a"] == ["GBP"]
        assert set(audit["fields_selected"]) <= set(audit["fields_considered"])
        assert audit["aggregations"]["current_loan_to_value"] == "weighted_average"
        assert audit["weights"]["current_loan_to_value"] == "current_outstanding_balance"
        assert audit["denominators"]["current_loan_to_value"]["portfolio_a"] \
            == pytest.approx(1000.0)
        for item in audit["fields_excluded"]:
            assert item["reason"], item
        assert audit["outcome"] == "success"

    def test_controlled_failure_still_carries_an_audit(self, bsr):
        result = _run(_frame(), "Compare direct_001 with acquired_009", bsr=bsr)
        assert result["audit"]["outcome"] == "controlled_failure"
        assert result["audit"]["bsr_version"] == bsr.version


# --------------------------------------------------------------------------- #
# The semantics resolver seam
# --------------------------------------------------------------------------- #
class TestSemanticsResolver:
    def test_resolver_attaches_governed_metadata_for_a_parsed_metric(self):
        context = semantics_context_resolver(
            "which portfolio has higher ltv",
            _Spec(metric="current_loan_to_value"), {})
        assert context["bsr_schema_version"] == 2
        entry = context["fields"]["metric"]
        assert entry["source_field"] == "current_loan_to_value"
        assert entry["directionality"] == "higher_is_worse"
        assert entry["weight_field"] == "current_outstanding_balance"

    def test_resolver_is_empty_for_ungoverned_terms(self):
        context = semantics_context_resolver("hello", _Spec(), {})
        assert "fields" not in context

    def test_resolver_never_raises_when_the_registry_is_unreadable(self, monkeypatch):
        monkeypatch.setenv("TRAKT_BUSINESS_SEMANTICS_REGISTRY", "/nonexistent.yaml")
        assert semantics_context_resolver("q", _Spec(), {}) == {}
