"""Concentration Analysis — the pure governed workflow.

Covers the workflow package (``mi_workflows``): recognition predicates (and
the ownership boundaries they defer to), single-scope resolution, one-
reporting-date enforcement, Business Semantics Registry dimension governance
(role, the governed ``concentration`` category, asset applicability, scale
alignment on multi-portfolio scopes), the shared calculation engine's ranked
distributions (count and exposure bases, deterministic ordering, cumulative
shares, top-N), explicit unknown handling, the mixed-currency guard,
single-name concentration over governed identifiers, evidence, audit, and the
observational-summary discipline. The API adapter and recogniser registration
are covered separately in
``mi_agent_api/tests/test_concentration_analysis_route.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_workflows import concentration_analysis as ca
from mi_workflows import engine
from mi_workflows.semantics import load_business_semantics

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bsr():
    return load_business_semantics(REPO / "config" / "business_semantics_registry.yaml")


def _frame(**overrides) -> pd.DataFrame:
    """Two governed portfolios, four loans each, one reporting date.

    Exposure totals are chosen for hand-checkable shares: the book totals
    2,200, of which london holds 900, south 750, north 550.
    """
    base = {
        "source_portfolio_id": ["direct_001"] * 4 + ["acquired_001"] * 4,
        "source_portfolio_type": ["direct"] * 4 + ["acquired"] * 4,
        "reporting_date": ["2026-06-30"] * 8,
        "loan_identifier": [f"L{i}" for i in range(8)],
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0,
                                        150.0, 250.0, 350.0, 450.0],
        "exposure_currency_denomination": ["GBP"] * 8,
        "product_type": ["lump", "lump", "draw", "draw",
                         "lump", "draw", "draw", None],
        "origination_channel": ["broker", "direct", "broker", "broker",
                                "direct", "broker", None, "broker"],
        "geographic_region_obligor": ["london", "london", "north", "south",
                                      "london", "north", "south", "london"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _run(df, question="Show product concentration", *, bsr, **kw):
    """Run the workflow the way the ROUTER runs it.

    The workflow no longer resolves a portfolio scope or a concept from the
    question after the route is claimed — the recogniser reads the question
    once, pre-claim, and the adapter hands the reading and the resolved scope
    in. This harness stands in for the router and does the same two steps. No
    assertion below changes; what changes is who reads the sentence.
    """
    from mi_agent import portfolio_lens as _lens_mod

    kw.setdefault("as_of", "2026-06-30")
    if "context_id" not in kw:
        kw["context_id"] = _lens_mod.context_id(_lens_mod.resolve_lens(question))
    kw.setdefault("reading", ca.read_question(question))
    return ca.run_concentration_analysis(df, question=question, bsr=bsr, **kw)


def _dimension(result, field):
    for d in result["dimension_results"]:
        if d["field"] == field:
            return d
    return None


class _Spec:
    def __init__(self, dimension=None, top_n=None, temporal_mode=None,
                 forecast_mode=None, risk_limit_query=None, bridge_query=None):
        self.dimension = dimension
        self.top_n = top_n
        self.temporal_mode = temporal_mode
        self.forecast_mode = forecast_mode
        self.risk_limit_query = risk_limit_query
        self.bridge_query = bridge_query


#: parse_meta for an explicitly named dimension (the single parser's shape).
_EXPLICIT = {"explicit_dimension_requested": True, "dimension_substituted": False}


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
class TestRecognition:
    @pytest.mark.parametrize("question", [
        "Show broker concentration",
        "Show product concentration",
        "What are the largest concentrations?",
        "How diversified is the portfolio?",
        "Show exposure by broker",
        "Show exposure by product",
        "Show distribution by product",
        "Show product mix",
        "Show originator mix",
        "Show geographic mix",
        "Show currency mix",
        "Show top exposures",
        "Show largest borrower exposures",
        "Show single name concentration",
        "Show the top 10 broker concentration",
        "Show broker concentration for the acquired book",
    ])
    def test_concentration_language_is_recognised(self, question):
        matched, reason = ca.is_concentration_question(question)
        assert matched, reason

    @pytest.mark.parametrize("question", [
        # period change — owned by the temporal routes
        "How has broker concentration changed since last month?",
        "Show product mix trend over time",
        # portfolio comparison — owned by portfolio_risk_comparison
        "Compare concentration across portfolios",
        "Compare product mix between direct_001 and acquired_001",
        # covenant / limits — owned by the risk-limit monitor
        "Show geographic concentration limits",
        "Which concentration limit is closest to breach?",
        "Are we within our concentration limits?",
        "What is the headroom on the broker limit?",
        # ITL3 geographic exposure — owned by the geo route
        "Show geographic concentration",
        "What is the largest geographic area concentration?",
        "Where is the book concentrated geographically?",
        "Which regions contain most exposure?",
        # forecast
        "Forecast broker concentration next quarter",
        # eligibility
        "Which loans are eligible for the warehouse?",
        # raw exports
        "Export the loan level data",
        # ranked stratifications — owned by the point-in-time executor
        "top 5 brokers by balance",
        "show the top 10 areas by exposure",
        "Show NNEG exposure by LTV bucket",
        # concentration indices — explicitly not implemented
        "What is the HHI?",
        "Calculate the Herfindahl index",
        # not concentration at all
        "balance by broker",
        "What is the average LTV?",
        "Summarise the portfolio",
    ])
    def test_questions_owned_elsewhere_are_rejected(self, question):
        matched, reason = ca.is_concentration_question(question)
        assert not matched, f"stole {question!r} ({reason})"

    def test_parsed_intents_owned_elsewhere_are_rejected(self):
        for spec, needle in [
            (_Spec(temporal_mode="compare"), "cross-period"),
            (_Spec(forecast_mode="extrapolation"), "forecast"),
            (_Spec(risk_limit_query=True), "risk-limit"),
            (_Spec(bridge_query=True), "bridge"),
        ]:
            matched, reason = ca.is_concentration_question(
                "Show broker concentration", spec)
            assert not matched
            assert needle in reason

    def test_single_name_kind_resolution(self):
        assert ca.requested_single_name_kind("Show top exposures") == "loan"
        assert ca.requested_single_name_kind("largest borrower exposures") == "borrower"
        assert ca.requested_single_name_kind("top 10 obligors") == "obligor"
        assert ca.requested_single_name_kind("single name concentration") == "loan"
        # "largest concentrations" is an overview, not a single-name ranking.
        assert ca.requested_single_name_kind("largest concentrations") is None
        assert ca.requested_single_name_kind("broker concentration") is None


# --------------------------------------------------------------------------- #
# Scope resolution — one governed scope
# --------------------------------------------------------------------------- #
class TestScopeResolution:
    def test_default_scope_is_total(self, bsr):
        result = _run(_frame(), bsr=bsr)
        assert result["available"] is True
        assert result["portfolio_scope"]["context_id"] == "total"
        assert result["portfolio_scope"]["row_count"] == 8

    def test_a_named_book_narrows_the_scope(self, bsr):
        result = _run(_frame(), "Show product concentration for the acquired book",
                      bsr=bsr)
        assert result["portfolio_scope"]["context_id"] == "acquired"
        assert result["portfolio_scope"]["row_count"] == 4
        # Shares are shares of the SCOPE: acquired book totals 1,200.
        product = _dimension(result, "product_type")
        draw = next(c for c in product["categories"] if c["category"] == "draw")
        assert draw["exposure_share"] == pytest.approx(600 / 1200)

    def test_a_workspace_context_id_is_honoured(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr,
                      context_id="direct_001")
        assert result["portfolio_scope"]["context_id"] == "direct_001"
        assert result["portfolio_scope"]["row_count"] == 4

    def test_an_unknown_cohort_is_a_controlled_failure_listing_governed_ids(self, bsr):
        result = _run(_frame(), "Show product concentration for acquired_009",
                      bsr=bsr)
        assert result["available"] is False
        assert "acquired_009" in result["reason"]
        assert "direct_001" in result["reason"]          # what IS available

    def test_an_empty_scope_is_a_controlled_failure(self, bsr):
        from mi_agent.portfolio_scope import registry_for_frame

        registry = registry_for_frame(_frame(), client_id="c")
        df = _frame()
        df = df[df["source_portfolio_id"] == "direct_001"]
        result = _run(df, "Show product concentration for acquired_001",
                      bsr=bsr, registry=registry)
        assert result["available"] is False
        assert "no rows in scope" in result["reason"]

    def test_a_tape_without_provenance_still_measures_the_whole_book(self, bsr):
        df = _frame().drop(columns=["source_portfolio_id",
                                    "source_portfolio_type"])
        result = _run(df, bsr=bsr)
        assert result["available"] is True
        assert result["portfolio_scope"]["row_count"] == 8


# --------------------------------------------------------------------------- #
# Reporting date — one date only
# --------------------------------------------------------------------------- #
class TestReportingDate:
    def test_the_single_date_is_reported(self, bsr):
        result = _run(_frame(), bsr=bsr)
        assert result["reporting_date"] == "2026-06-30"
        assert result["audit"]["reporting_date"] == "2026-06-30"

    def test_a_scope_spanning_dates_is_refused(self, bsr):
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
# Dimension governance — the registry decides
# --------------------------------------------------------------------------- #
class TestDimensionGovernance:
    def test_only_concentration_category_dimensions_are_considered(self, bsr):
        # account_status is a governed dimension but does NOT carry the
        # concentration category — it must never appear in an overview.
        entry = bsr.get("account_status")
        assert entry.analytical_role == "dimension"
        assert "concentration" not in entry.categories
        df = _frame(account_status=["performing"] * 8)
        result = _run(df, "How diversified is the portfolio?", bsr=bsr)
        assert _dimension(result, "account_status") is None
        assert "account_status" not in result["audit"]["dimensions_considered"]

    def test_measures_are_never_concentration_dimensions(self, bsr):
        result = _run(_frame(), "How diversified is the portfolio?", bsr=bsr)
        for field in result["audit"]["dimensions_selected"]:
            assert bsr.get(field).analytical_role == "dimension"
            assert "concentration" in bsr.get(field).categories

    def test_asset_specific_dimension_needs_the_declared_asset_class(self, bsr):
        entry = bsr.get("tenure")
        assert entry.asset_applicability == ("residential_mortgage",)
        df = _frame(tenure=["freehold", "leasehold"] * 4)
        result = _run(df, "Show tenure mix", bsr=bsr)
        excluded = {e["field"]: e for e in result["audit"]["dimensions_excluded"]}
        assert "tenure" in excluded
        assert "asset class" in excluded["tenure"]["reason"]

        with_class = _frame(tenure=["freehold", "leasehold"] * 4,
                            asset_class=["residential_mortgage"] * 8)
        result = _run(with_class, "Show tenure mix", bsr=bsr)
        assert _dimension(result, "tenure") is not None

    def test_scale_aligned_vocabulary_is_excluded_on_a_multi_portfolio_scope(self, bsr):
        # erm_product_type is an originator vocabulary the registry marks
        # requires_scale_alignment: on a scope spanning two portfolios its
        # categories would mix vocabularies, so it is excluded with the reason
        # recorded — and the governed product_type answers instead.
        assert (bsr.get("erm_product_type").portfolio_comparability
                == "requires_scale_alignment")
        df = _frame(erm_product_type=["Lifetime"] * 8,
                    asset_class=["equity_release"] * 8)
        result = _run(df, "Show product concentration", bsr=bsr,
                      spec=_Spec(dimension="erm_product_type"),
                      parse_meta=_EXPLICIT)
        assert _dimension(result, "erm_product_type") is None
        assert _dimension(result, "product_type") is not None
        excluded = {e["field"]: e for e in result["audit"]["dimensions_excluded"]}
        # THE REASON MUST NAME THE ORIGINATOR-SPECIFIC CHARACTER of the
        # categories. This matched the literal word "vocabulary", which was a
        # proxy for that fact; the sentence now says "the registry declares
        # this dimension's categories originator-specific" and adds what the
        # limit actually is — the METHODOLOGY's, not arithmetic's, because the
        # same dimension is legitimately aggregated by the share and ranking
        # paths. Every structural guarantee this test makes is unchanged and
        # still asserted above and below: erm_product_type excluded,
        # product_type answering instead, the exclusion recorded in the audit,
        # and a limitation emitted.
        assert "originator" in excluded["erm_product_type"]["reason"]
        assert any("erm_product_type" in note for note in result["limitations"])

    def test_scale_aligned_vocabulary_is_allowed_within_one_portfolio(self, bsr):
        df = _frame(erm_product_type=["Lifetime"] * 8,
                    asset_class=["equity_release"] * 8)
        result = _run(df, "Show product concentration for direct_001", bsr=bsr,
                      spec=_Spec(dimension="erm_product_type"),
                      parse_meta=_EXPLICIT)
        assert result["audit"]["mode"] == "requested_dimension"
        assert _dimension(result, "erm_product_type") is not None

    def test_a_parser_default_dimension_is_not_treated_as_a_request(self, bsr):
        # The deterministic parser substitutes a default dimension for generic
        # concentration questions; the workflow must not read that default as
        # the user's ask.
        result = _run(_frame(), "What are the largest concentrations?", bsr=bsr,
                      spec=_Spec(dimension="geographic_region_obligor"),
                      parse_meta={"explicit_dimension_requested": False,
                                  "dimension_substituted": False})
        assert result["audit"]["mode"] == "overview"

    def test_an_ungoverned_dimension_is_explained(self, bsr):
        df = _frame(bespoke_grouping=["a"] * 8)
        result = _run(df, "Show exposure by product", bsr=bsr,
                      spec=_Spec(dimension="bespoke_grouping"),
                      parse_meta=_EXPLICIT)
        assert any("not governed" in note for note in result["limitations"])

    def test_nothing_measurable_is_a_controlled_failure(self, bsr):
        df = pd.DataFrame({
            "reporting_date": ["2026-06-30"] * 4,
            "current_outstanding_balance": [1.0, 2.0, 3.0, 4.0],
        })
        result = _run(df, "How diversified is the portfolio?", bsr=bsr)
        assert result["available"] is False
        assert "concentration cannot be measured" in result["reason"]
        assert result["audit"]["outcome"] == "controlled_failure"


# --------------------------------------------------------------------------- #
# Core calculations — the shared engine's ranked distributions
# --------------------------------------------------------------------------- #
class TestCalculations:
    def test_exposure_shares_ranks_and_cumulative_are_hand_checkable(self, bsr):
        result = _run(_frame(), "Show geographic mix", bsr=bsr)
        geo = _dimension(result, "geographic_region_obligor")
        assert geo["basis"] == "exposure"
        assert geo["population"] == 8
        assert geo["total_exposure"] == pytest.approx(2200.0)
        cats = geo["categories"]
        # london 900, south 750, north 550 of 2,200 — ranked by exposure.
        assert [(c["rank"], c["category"]) for c in cats] == [
            (1, "london"), (2, "south"), (3, "north")]
        assert cats[0]["exposure_share"] == pytest.approx(900 / 2200)
        assert cats[1]["cumulative_share"] == pytest.approx(1650 / 2200)
        assert cats[2]["cumulative_share"] == pytest.approx(1.0)
        # Count is returned alongside exposure on every category.
        assert cats[0]["count"] == 4
        assert cats[0]["count_share"] == pytest.approx(4 / 8)

    def test_count_concentration_is_always_available(self, bsr):
        df = _frame().drop(columns=["current_outstanding_balance"])
        result = _run(df, "Show geographic mix", bsr=bsr)
        assert result["concentration_basis"] == "count"
        geo = _dimension(result, "geographic_region_obligor")
        assert geo["basis"] == "count"
        assert geo["categories"][0]["category"] == "london"
        assert geo["categories"][0]["count_share"] == pytest.approx(4 / 8)
        assert geo["categories"][0]["exposure"] is None
        assert any("count concentration only" in note
                   for note in result["limitations"])

    def test_ordering_is_deterministic_including_ties(self, bsr):
        df = _frame(current_outstanding_balance=[100.0] * 8,
                    geographic_region_obligor=["b", "b", "a", "a",
                                               "c", "c", "a", "b"])
        result = _run(df, "Show geographic mix", bsr=bsr)
        geo = _dimension(result, "geographic_region_obligor")
        # a and b tie on 3 loans / 300 exposure: alphabetical break, c last.
        assert [c["category"] for c in geo["categories"]] == ["a", "b", "c"]

    def test_top_n_is_honoured(self, bsr):
        result = _run(_frame(), "Show the top 2 regions concentration", bsr=bsr,
                      spec=_Spec(top_n=2))
        geo = _dimension(result, "geographic_region_obligor")
        assert geo["top_shares"]["top_2"] == pytest.approx(1650 / 2200)
        assert geo["top_shares"]["top_5"] == pytest.approx(1.0)
        assert result["audit"]["top_n"] == 2

    def test_deterministic_output(self, bsr):
        one = _run(_frame(), "How diversified is the portfolio?", bsr=bsr)
        two = _run(_frame(), "How diversified is the portfolio?", bsr=bsr)
        assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


# --------------------------------------------------------------------------- #
# Unknown values — always explicit, never dropped
# --------------------------------------------------------------------------- #
class TestUnknowns:
    def test_unknowns_are_explicit_in_counts_and_exposure(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr)
        product = _dimension(result, "product_type")
        unknown = product["unknown"]
        # One null product (L7, 450 of 2,200).
        assert unknown["count"] == 1
        assert unknown["count_share"] == pytest.approx(1 / 8)
        assert unknown["exposure"] == pytest.approx(450.0)
        assert unknown["exposure_share"] == pytest.approx(450 / 2200)
        # Unknowns keep the denominator honest: known shares + unknown = 1.
        known = sum(c["exposure_share"] for c in product["categories"])
        assert known + unknown["exposure_share"] == pytest.approx(1.0)
        # The unknown bucket is never ranked as a category.
        assert all(c["category"] != engine.UNKNOWN_CATEGORY
                   for c in product["categories"])

    def test_unknowns_are_disclosed_in_the_summary(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr)
        assert any("Unknown product type" in line for line in result["summary"])

    def test_a_fully_known_dimension_reports_zero_unknowns(self, bsr):
        result = _run(_frame(), "Show geographic mix", bsr=bsr)
        geo = _dimension(result, "geographic_region_obligor")
        assert geo["unknown"]["count"] == 0
        assert geo["unknown"]["exposure_share"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Mixed currencies — the shared guard, no FX
# --------------------------------------------------------------------------- #
class TestMixedCurrency:
    def test_a_mixed_scope_suppresses_exposure_and_keeps_counts(self, bsr):
        df = _frame()
        df.loc[df.index[:2], "exposure_currency_denomination"] = "USD"
        result = _run(df, "Show product concentration", bsr=bsr)
        assert result["available"] is True
        assert result["concentration_basis"] == "count"
        product = _dimension(result, "product_type")
        assert product["basis"] == "count"
        assert all(c["exposure"] is None for c in product["categories"])
        assert any("no FX conversion" in note for note in result["limitations"])
        assert result["audit"]["exposure_concentration_allowed"] is False
        assert result["audit"]["currencies"] == ["GBP", "USD"]

    def test_a_single_currency_scope_defaults_to_the_exposure_basis(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr)
        assert result["concentration_basis"] == "exposure"
        assert result["audit"]["exposure_concentration_allowed"] is True

    def test_a_narrowed_scope_escapes_the_other_books_currencies(self, bsr):
        # Only the acquired book is USD: the direct scope stays internally
        # consistent, so exposure concentration continues there.
        df = _frame()
        df.loc[df.source_portfolio_id == "acquired_001",
               "exposure_currency_denomination"] = "USD"
        result = _run(df, "Show product concentration for direct_001", bsr=bsr)
        assert result["concentration_basis"] == "exposure"

    def test_no_currency_column_discloses_the_book_assumption(self, bsr):
        df = _frame().drop(columns=["exposure_currency_denomination"])
        result = _run(df, "Show product concentration", bsr=bsr)
        assert result["concentration_basis"] == "exposure"
        assert any("single-currency book assumption" in w
                   for w in result["warnings"])


# --------------------------------------------------------------------------- #
# Single-name concentration — governed identifiers only
# --------------------------------------------------------------------------- #
class TestSingleName:
    def test_top_exposures_rank_loans_by_the_governed_identifier(self, bsr):
        result = _run(_frame(), "Show top exposures", bsr=bsr)
        assert result["audit"]["mode"] == "single_name"
        assert result["dimension_results"] == []
        sn = result["single_name_results"][0]
        assert sn["kind"] == "loan"
        assert sn["field"] == "loan_identifier"
        assert sn["distinct_names"] == 8
        assert sn["categories"][0]["category"] == "L7"       # 450 of 2,200
        assert sn["categories"][0]["exposure_share"] == pytest.approx(450 / 2200)

    def test_borrower_concentration_uses_the_governed_borrower_identifier(self, bsr):
        df = _frame(borrower_identifier=["B1", "B1", "B2", "B2",
                                         "B3", "B3", "B3", "B4"])
        result = _run(df, "Show largest borrower exposures", bsr=bsr)
        sn = result["single_name_results"][0]
        assert sn["kind"] == "borrower"
        assert sn["field"] == "borrower_identifier"
        # B3 holds 150+250+350 = 750 of 2,200.
        assert sn["categories"][0]["category"] == "B3"
        assert sn["categories"][0]["exposure_share"] == pytest.approx(750 / 2200)
        assert sn["categories"][0]["count"] == 3

    def test_identifier_precedence_is_declared_not_heuristic(self, bsr):
        df = _frame(borrower_identifier=["B1"] * 8, borrower_1_id=["X1"] * 8)
        result = _run(df, "Show largest borrower exposures", bsr=bsr)
        assert result["single_name_results"][0]["field"] == "borrower_identifier"
        considered = result["audit"]["single_name_identifiers"]
        assert [c["field"] for c in considered] == ["borrower_identifier",
                                                    "borrower_1_id"]

    def test_a_missing_identifier_is_a_controlled_failure(self, bsr):
        df = _frame().drop(columns=["loan_identifier"])
        result = _run(df, "Show top exposures", bsr=bsr)
        assert result["available"] is False
        assert "no governed loan identifier" in result["reason"]

    def test_the_listing_is_bounded_with_an_explicit_remainder(self, bsr):
        result = _run(_frame(), "Show the top 3 exposures", bsr=bsr,
                      spec=_Spec(top_n=3))
        sn = result["single_name_results"][0]
        assert sn["listed"] == 3
        assert sn["remainder"]["names"] == 5
        # Top 3: 450 + 400 + 350 of 2,200; the remainder is the rest.
        assert sn["remainder"]["share"] == pytest.approx(1 - 1200 / 2200)


# --------------------------------------------------------------------------- #
# Summary discipline, evidence, audit
# --------------------------------------------------------------------------- #
class TestSummaryEvidenceAudit:
    def test_summary_is_observational_never_a_judgement(self, bsr):
        for question in ("Show product concentration",
                         "How diversified is the portfolio?",
                         "Show top exposures"):
            result = _run(_frame(), question, bsr=bsr)
            text = " ".join(result["summary"]).lower()
            for forbidden in ("excessive", "breach", "appetite", "limit",
                              "acceptable", "compliant", "insufficient",
                              "too concentrated", "well diversified"):
                assert forbidden not in text, (question, text)

    def test_summary_states_largest_top_n_and_unknowns(self, bsr):
        result = _run(_frame(), "Show geographic mix", bsr=bsr)
        text = " ".join(result["summary"])
        assert "Largest geographic region obligor exposure is london" in text
        result = _run(_frame(), "Show the top 2 regions concentration",
                      bsr=bsr, spec=_Spec(top_n=2))
        text = " ".join(result["summary"])
        assert "Top 2" in text and "75.0%" in text        # 1,650 of 2,200

    def test_no_concentration_indices_anywhere_in_the_contract(self, bsr):
        result = _run(_frame(), "How diversified is the portfolio?", bsr=bsr)
        text = json.dumps(result).lower()
        for forbidden in ("hhi", "herfindahl", "gini", "concentration_index",
                          "concentration index"):
            assert forbidden not in text

    def test_evidence_discloses_populations_and_sources(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr)
        evidence = next(e for e in result["evidence"]
                        if e["field"] == "product_type")
        assert evidence["kind"] == "concentration_population"
        assert evidence["population"] == 8
        assert evidence["total_exposure"] == pytest.approx(2200.0)
        assert evidence["unknown_count"] == 1
        assert evidence["source"] == "funded canonical tape"

    def test_single_name_evidence_discloses_the_listing_bound(self, bsr):
        result = _run(_frame(), "Show top exposures", bsr=bsr)
        evidence = next(e for e in result["evidence"]
                        if e["kind"] == "single_name_population")
        assert evidence["distinct_names"] == 8
        assert evidence["listed"] == 8

    def test_audit_records_the_governed_decision_trail(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr)
        audit = result["audit"]
        assert audit["workflow_id"] == ca.WORKFLOW_ID
        assert audit["bsr_version"] == bsr.version
        assert audit["bsr_schema_version"] == bsr.schema_version
        assert audit["calculation_version"] == engine.CALCULATION_VERSION
        assert audit["dataset"] == "funded"
        assert audit["reporting_date"] == "2026-06-30"
        assert audit["portfolio_scope"]["context_id"] == "total"
        assert audit["concentration_basis"] == "exposure"
        assert audit["currencies"] == ["GBP"]
        assert set(audit["dimensions_selected"]) <= set(audit["dimensions_considered"])
        assert audit["denominators"]["product_type"]["count"] == 8
        assert audit["denominators"]["product_type"]["exposure"] == pytest.approx(2200.0)
        for item in audit["dimensions_excluded"]:
            assert item["reason"], item
        assert audit["outcome"] == "success"

    def test_a_controlled_failure_still_carries_an_audit(self, bsr):
        result = _run(_frame(), "Show product concentration for acquired_009",
                      bsr=bsr)
        assert result["audit"]["outcome"] == "controlled_failure"
        assert result["audit"]["bsr_version"] == bsr.version

    def test_the_result_contract_keys_are_stable(self, bsr):
        result = _run(_frame(), "Show product concentration", bsr=bsr)
        assert set(result.keys()) == {
            "workflow_id", "available", "reason", "portfolio_scope",
            "reporting_date", "concentration_basis", "dimension_results",
            "single_name_results", "warnings", "limitations", "summary",
            "evidence", "audit"}


# --------------------------------------------------------------------------- #
# The shared engine's ranked-distribution primitive
# --------------------------------------------------------------------------- #
class TestRankedDistributionPrimitive:
    def _df(self):
        return pd.DataFrame({
            "cat": ["a", "b", "a", None, "c", "b", "a", "c"],
            "bal": [10.0, 40.0, 20.0, 5.0, 15.0, 30.0, 10.0, 15.0],
        })

    def test_ranks_cumulative_and_unknowns(self):
        ranked = engine.ranked_distribution(self._df(), "cat",
                                            exposure_field="bal")
        # b=70, a=40, c=30 of 145 total; unknown 5.
        assert [b.category for b in ranked.buckets] == ["b", "a", "c"]
        assert ranked.buckets[0].rank == 1
        assert ranked.buckets[0].exposure_share == pytest.approx(70 / 145)
        assert ranked.buckets[1].cumulative_share == pytest.approx(110 / 145)
        assert ranked.buckets[2].cumulative_share == pytest.approx(140 / 145)
        assert ranked.unknown_count == 1
        assert ranked.unknown_exposure_share == pytest.approx(5 / 145)
        assert ranked.top_share(2) == pytest.approx(110 / 145)
        assert ranked.top_share(99) == pytest.approx(140 / 145)

    def test_count_basis_ranking(self):
        ranked = engine.ranked_distribution(self._df(), "cat",
                                            basis=engine.BASIS_COUNT)
        # a=3, b=2, c=2 of 8 rows: count ties break alphabetically.
        assert [b.category for b in ranked.buckets] == ["a", "b", "c"]
        assert ranked.buckets[0].count_share == pytest.approx(3 / 8)
        assert ranked.buckets[1].cumulative_share == pytest.approx(5 / 8)

    def test_exposure_basis_needs_positive_exposure(self):
        df = self._df().assign(bal=0.0)
        assert engine.ranked_distribution(df, "cat", exposure_field="bal") is None
        df = self._df().drop(columns=["bal"])
        assert engine.ranked_distribution(df, "cat", exposure_field="bal") is None

    def test_missing_field_or_unknown_basis_is_none(self):
        assert engine.ranked_distribution(self._df(), "nope",
                                          exposure_field="bal") is None
        assert engine.ranked_distribution(self._df(), "cat",
                                          basis="weighted") is None

    def test_the_ranking_never_disagrees_with_the_distribution(self):
        df = self._df()
        dist = engine.distribution(df, "cat", exposure_field="bal")
        ranked = engine.ranked_distribution(df, "cat", exposure_field="bal")
        for bucket in ranked.buckets:
            plain = dist.bucket(bucket.category)
            assert bucket.count == plain.count
            assert bucket.exposure_share == plain.exposure_share
