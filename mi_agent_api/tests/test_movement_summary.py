"""Tests for the governed composite MI answers.

``mi_agent_api.movement_summary`` composes the existing evolution and bridge
services into a current-period portfolio summary and a month-on-month movement.
These tests cover the composition and the wording rules against small synthetic
frames — no demonstration build required.

The routing tests matter as much as the service tests: the two new routes sit in
front of the compare / evolution / bridge routes, so they must fire on exactly the
intents they claim and defer on everything else.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import chat_routing, movement_summary as ms


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def _frame(*, n: int, balance: float, ltv: float, rate: float, age: float,
           region_split: dict, origination: str, portfolio_split: dict) -> pd.DataFrame:
    """A small funded frame with the columns the services read."""
    rows = []
    regions = []
    for region, share in region_split.items():
        regions.extend([region] * int(round(n * share)))
    regions = (regions + [next(iter(region_split))] * n)[:n]
    portfolios = []
    for pid, share in portfolio_split.items():
        portfolios.extend([pid] * int(round(n * share)))
    portfolios = (portfolios + [next(iter(portfolio_split))] * n)[:n]
    for i in range(n):
        rows.append({
            "loan_identifier": f"L{i:05d}",
            "current_outstanding_balance": balance / n,
            "current_loan_to_value": ltv,
            "current_interest_rate": rate,
            "youngest_borrower_age": age,
            "collateral_geography": regions[i],
            "geographic_region_obligor": "TLJ12",
            "origination_date": origination,
            "source_portfolio_id": portfolios[i],
            "source_portfolio_label": f"{portfolios[i].upper()} Book",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def frames(monkeypatch):
    """Two periods: balances up 10%, LTV stable, age up, South East leading."""
    prior = _frame(n=100, balance=1_000_000.0, ltv=0.40, rate=6.0, age=71.0,
                   region_split={"South East": 0.5, "London": 0.3, "Wales": 0.2},
                   origination="2020-01-15",
                   portfolio_split={"alp_origination": 0.7, "alp_acquired": 0.3})
    current = _frame(n=100, balance=1_100_000.0, ltv=0.41, rate=6.0, age=71.2,
                     region_split={"South East": 0.6, "London": 0.25, "Wales": 0.15},
                     origination="2026-06-10",
                     portfolio_split={"alp_origination": 0.7, "alp_acquired": 0.3})
    built = [
        {"run_id": "2026-05-31", "reporting_date": "2026-05-31", "df": prior,
         "source": "blob://x/2026-05-31/platform_canonical_typed.csv"},
        {"run_id": "2026-06-30", "reporting_date": "2026-06-30", "df": current,
         "source": "blob://x/2026-06-30/platform_canonical_typed.csv"},
    ]
    from mi_agent_api import evolution as evolution_mod
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: [dict(f) for f in built])
    return built


# --------------------------------------------------------------------------- #
# Portfolio summary
# --------------------------------------------------------------------------- #
class TestPortfolioSummary:
    def test_returns_the_current_period_position(self, frames):
        out = ms.portfolio_summary("blob://x", "client", to_run_id=None)
        assert out["available"] is True
        assert out["period"] == "2026-06"
        assert out["reportingDate"] == "2026-06-30"
        assert out["metrics"]["loan_count"] == 100
        assert out["metrics"]["funded_balance"] == pytest.approx(1_100_000.0)

    def test_normalises_ltv_to_percentage_points(self, frames):
        out = ms.portfolio_summary("blob://x", "client")
        # The prepared dataset stores LTV as a 0..1 ratio; the answer states points.
        assert out["metrics"]["wa_ltv_points"] == pytest.approx(41.0)

    def test_ranks_the_largest_regional_exposures(self, frames):
        out = ms.portfolio_summary("blob://x", "client")
        regions = [r["region"] for r in out["topRegions"]]
        assert regions[0] == "South East"
        assert sum(r["share"] for r in out["topRegions"]) == pytest.approx(1.0, abs=1e-6)

    def test_lists_the_source_portfolios_present(self, frames):
        out = ms.portfolio_summary("blob://x", "client")
        assert {c["id"] for c in out["cohorts"]} == {"alp_origination", "alp_acquired"}
        assert sum(out["cohortBalances"].values()) == pytest.approx(1_100_000.0)

    def test_reports_unavailable_rather_than_guessing(self, monkeypatch):
        from mi_agent_api import evolution as evolution_mod
        monkeypatch.setattr(evolution_mod, "funded_frames", lambda *a, **k: [])
        out = ms.portfolio_summary("blob://x", "client")
        assert out["available"] is False
        assert out["reason"]


# --------------------------------------------------------------------------- #
# Period movement
# --------------------------------------------------------------------------- #
class TestPeriodMovement:
    def test_computes_the_month_on_month_deltas(self, frames):
        out = ms.period_movement("blob://x", "client")
        assert out["available"] is True
        assert out["priorPeriod"] == "2026-05"
        assert out["currentPeriod"] == "2026-06"
        assert out["delta"]["funded_balance"] == pytest.approx(100_000.0)
        assert out["delta"]["avg_borrower_age"] == pytest.approx(0.2, abs=1e-6)

    def test_cohort_movements_sum_to_the_consolidated_movement(self, frames):
        out = ms.period_movement("blob://x", "client")
        total = sum(c["delta"] for c in out["cohortMovements"])
        assert total == pytest.approx(out["delta"]["funded_balance"], abs=0.05)
        assert out["reconciliation"]["cohort_residual"] == pytest.approx(0.0, abs=0.05)

    def test_needs_two_periods(self, monkeypatch, frames):
        from mi_agent_api import evolution as evolution_mod
        monkeypatch.setattr(evolution_mod, "funded_frames", lambda *a, **k: [frames[0]])
        out = ms.period_movement("blob://x", "client")
        assert out["available"] is False
        assert "two funded reporting periods" in out["reason"]

    def test_completions_evidence_is_measured_not_assumed(self, frames):
        """The completion wording must be backed by loans originated in-month."""
        out = ms.period_movement("blob://x", "client")
        # Every current-period loan in this fixture was originated in June 2026.
        assert out["completionsBalanceInMonth"] == pytest.approx(1_100_000.0)
        assert out["completionsEvidenced"] is True

    def test_completions_are_not_claimed_when_nothing_completed(self, monkeypatch):
        prior = _frame(n=50, balance=1_000_000.0, ltv=0.40, rate=6.0, age=71.0,
                       region_split={"South East": 1.0}, origination="2015-01-01",
                       portfolio_split={"alp_acquired": 1.0})
        current = _frame(n=50, balance=1_050_000.0, ltv=0.41, rate=6.0, age=71.1,
                         region_split={"South East": 1.0}, origination="2015-01-01",
                         portfolio_split={"alp_acquired": 1.0})
        from mi_agent_api import evolution as evolution_mod
        monkeypatch.setattr(evolution_mod, "funded_frames", lambda *a, **k: [
            {"run_id": "2026-05-31", "reporting_date": "2026-05-31", "df": prior},
            {"run_id": "2026-06-30", "reporting_date": "2026-06-30", "df": current},
        ])
        out = ms.period_movement("blob://x", "client")
        assert out["completionsBalanceInMonth"] == 0.0
        assert out["completionsEvidenced"] is False


# --------------------------------------------------------------------------- #
# Wording rules
# --------------------------------------------------------------------------- #
class TestWording:
    @pytest.mark.parametrize("delta,expected", [
        (0.28, "remained broadly stable at"),
        (-0.28, "remained broadly stable at"),
        (1.4, "rose to"),
        (-1.4, "fell to"),
        (None, "is not available at"),
    ])
    def test_ltv_clause(self, delta, expected):
        assert ms.ltv_descriptor(delta) == expected

    @pytest.mark.parametrize("delta,expected", [
        (0.06, "increased slightly to"),
        (-0.06, "decreased slightly to"),
        (0.01, "was unchanged at"),
        (1.2, "increased to"),
        (-1.2, "decreased to"),
    ])
    def test_age_clause(self, delta, expected):
        assert ms.age_descriptor(delta) == expected

    def test_every_clause_carries_its_own_preposition(self):
        """The caller writes "{metric} {clause} {value}", so each must be grammatical."""
        for clause in (ms.ltv_descriptor(0.1), ms.ltv_descriptor(2.0),
                       ms.age_descriptor(0.1), ms.age_descriptor(0.0)):
            assert clause.split()[-1] in ("at", "to")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
class TestRouting:
    @pytest.mark.parametrize("question", [
        "Summarise the portfolio.",
        "summarise the portfolio",
        "Give me a portfolio summary",
        "overview of the book",
    ])
    def test_summary_intent_is_recognised(self, question):
        assert chat_routing._is_portfolio_summary(question) is True

    @pytest.mark.parametrize("question", [
        "summarise the portfolio by region",     # a stratification
        "summarise what has changed vs the prior month",  # a movement question
        "balance by ltv bucket",
        "compare October and November funded balance",
    ])
    def test_summary_intent_defers(self, question):
        assert chat_routing._is_portfolio_summary(question) is False

    @pytest.mark.parametrize("question", [
        "What has changed versus the prior month?",
        "what's changed vs last month",
        "what moved versus the previous period",
        "how has the portfolio changed month on month",
    ])
    def test_movement_intent_is_recognised(self, question):
        assert chat_routing._is_period_movement(question) is True

    @pytest.mark.parametrize("question", [
        "What has changed?",                       # no period reference
        "compare October and November",            # the compare route owns this
        "funded balance over time",                # the evolution route owns this
        "funded balance bridge by region",         # the bridge route owns this
        "show me the prior month balance",         # not a change question
    ])
    def test_movement_intent_defers(self, question):
        assert chat_routing._is_period_movement(question) is False

    def test_the_movement_route_produces_a_reconciling_answer(self, frames):
        """CONVERSION 2 changed this test's CALLING CONVENTION, by decision.

        The same change, for the same reason, as Conversion 1 made to
        `test_the_summary_route_names_the_governed_metrics` below. The route now
        takes BOTH its semantic inputs — the population and the comparison
        window — from the interpretation contract, so the contract is a required
        input and a caller that supplies none gets a deferral. That is
        deliberate: keeping the lens-resolved path as a fallback would leave
        `_resolve_lens` and a second reading of the question reachable exactly
        when the contract failed, which is the worst moment for two owners to
        disagree.

        In production the handler has one caller, the recogniser registry, and
        it always supplies a contract. This test called the handler directly
        with none, which is a convention production no longer uses, so it now
        supplies one as production does. The assertions below are unchanged:
        same route, same movement, same South East attribution, same artifacts.
        """
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent.llm_query_parser import _deterministic_parse
        from question_interpretation import projection
        from pathlib import Path
        semantics_path = (Path(__file__).resolve().parents[2]
                          / "mi_agent" / "mi_semantics_field_registry.yaml")
        semantics = load_mi_semantics(semantics_path)
        question = "What has changed versus the prior month?"
        spec, _ = _deterministic_parse(question, semantics)
        interpretation = projection.project(question, semantics=semantics)
        out = chat_routing._route_period_movement(
            question, spec, spec.to_dict(), client_id="client", run_id=None,
            output_root="blob://x", portfolio_id="client/2026-06-30", as_of=None,
            interpretation=interpretation)
        assert out["ok"] is True
        assert out["metadata"]["route"] == "period_movement"
        assert "increased" in out["answer"]
        assert "South East" in out["answer"]
        # A KPI artifact plus the regional and cohort evidence.
        kinds = [a["type"] for a in out["artifacts"]]
        assert "kpi" in kinds and "chart" in kinds and "table" in kinds

    def test_the_summary_route_names_the_governed_metrics(self, frames):
        """CONVERSION 1 changed this test's CALLING CONVENTION, by decision.

        The route now plans its population from the interpretation contract, so
        the contract is a required input and a caller that supplies none gets a
        deferral — deliberately, because keeping the lens-resolved path as a
        fallback would leave a second population owner reachable exactly when
        the first one failed.

        In production the handler has one caller, the recogniser registry, and
        it always supplies one: measured at 54 of 54 owned renders. This test
        called the handler directly with no contract, which is a convention
        production no longer uses, so it now supplies one as production does.
        The assertions below are unchanged.
        """
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent.llm_query_parser import _deterministic_parse
        from question_interpretation import projection
        from pathlib import Path
        semantics_path = (Path(__file__).resolve().parents[2]
                          / "mi_agent" / "mi_semantics_field_registry.yaml")
        semantics = load_mi_semantics(semantics_path)
        question = "Summarise the portfolio."
        spec, _ = _deterministic_parse(question, semantics)
        interpretation = projection.project(question, semantics=semantics)
        out = chat_routing._route_portfolio_summary(
            question, spec, spec.to_dict(), client_id="client", run_id=None,
            output_root="blob://x", portfolio_id="client/2026-06-30", as_of=None,
            interpretation=interpretation)
        assert out is not None
        assert out["metadata"]["route"] == "portfolio_summary"
        assert "100 loans" in out["answer"]
        assert "41.0%" in out["answer"]
        assert "South East" in out["answer"]

    def test_the_summary_route_defers_when_no_periods_are_available(self, monkeypatch):
        from mi_agent_api import evolution as evolution_mod
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent.llm_query_parser import _deterministic_parse
        from pathlib import Path
        monkeypatch.setattr(evolution_mod, "funded_frames", lambda *a, **k: [])
        semantics_path = (Path(__file__).resolve().parents[2]
                          / "mi_agent" / "mi_semantics_field_registry.yaml")
        semantics = load_mi_semantics(semantics_path)
        spec, _ = _deterministic_parse("Summarise the portfolio.", semantics)
        out = chat_routing._route_portfolio_summary(
            "Summarise the portfolio.", spec, spec.to_dict(), client_id="c",
            run_id=None, output_root="blob://x", portfolio_id=None, as_of=None)
        assert out is None, "must defer to the existing point-in-time path"


# --------------------------------------------------------------------------- #
# Region column resolution
# --------------------------------------------------------------------------- #
class TestRegionColumn:
    def test_prefers_the_readable_label_over_the_regulatory_code(self):
        df = pd.DataFrame({
            "collateral_geography": ["South East", "London"],
            "geographic_region_obligor": ["TLJ12", "TLI35"],
        })
        assert ms._region_column(df) == "collateral_geography"

    def test_falls_back_to_the_code_when_no_label_is_present(self):
        df = pd.DataFrame({"geographic_region_obligor": ["TLJ12", "TLI35"]})
        assert ms._region_column(df) == "geographic_region_obligor"

    def test_skips_a_present_but_empty_column(self):
        df = pd.DataFrame({
            "collateral_geography": [None, None],
            "geographic_region_obligor": ["TLJ12", "TLI35"],
        })
        assert ms._region_column(df) == "geographic_region_obligor"
