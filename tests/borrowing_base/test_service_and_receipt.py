"""The MI seam and the audit receipt.

THE SEAM. The borrowing base is exposed as a stable, deterministic, named set
of measures that can be invoked WITHOUT Streamlit, FastAPI, React or any
dashboard module in the process. That is what makes a later MI Query Agent
sprint a registration rather than a rewrite. These tests assert exactly that
and nothing about MI Query itself — the MI Query Agent does not answer
borrowing-base questions today, and nothing here claims it does.

THE RECEIPT. A reviewer who was not there must be able to reproduce why Trakt
reported a particular borrowing base: the terms, their version, the population,
each equation's inputs, the concentration results as they stood, and every
assumption in play.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from mi_agent.borrowing_base.eligibility import derive_eligibility
from mi_agent.borrowing_base.models import NOT_CALCULABLE
from mi_agent.borrowing_base.service import (
    MEASURE_IDS,
    evaluate,
    measure_definitions,
)

from .conftest import facility, frame, production_facility

#: A £100,000,000 eligible book: gross base £103,000,000, well under the cap.
BOOK = [{"loan_id": "L0", "current_outstanding_balance": 100_000_000.0}]

CONCENTRATION = [
    {"testId": "t_east", "displayName": "East of England", "status": "breach",
     "currentValue": 30.0, "threshold": 25.0, "unit": "percent",
     "utilization": 120.0, "breachAmount": 5.0, "headroom": -5.0,
     "denominatorValue": 100_000_000.0, "numeratorValue": 30_000_000.0,
     "loansInNumerator": 12, "denominatorBasis": "current_balance",
     "dataStatus": "ok", "category": "geography", "operator": "max",
     "metricId": "geo_region_share", "missingFields": [],
     "population": "eligible_mortgage_loans",
     "provenance": {"sourceText": "UKH East of England [25]%"}},
    {"testId": "t_scot", "displayName": "Scotland", "status": "pass",
     "currentValue": 9.0, "threshold": 10.0, "unit": "percent",
     "utilization": 90.0, "breachAmount": None, "headroom": 1.0,
     "denominatorValue": 100_000_000.0, "numeratorValue": 9_000_000.0,
     "loansInNumerator": 4, "denominatorBasis": "current_balance",
     "dataStatus": "ok", "category": "geography", "operator": "max",
     "metricId": "geo_region_share", "missingFields": [],
     "population": "eligible_mortgage_loans",
     "provenance": {"sourceText": "UKM Scotland [10]%"}},
]


def snapshot(**over):
    fac = over.pop("facility", None) or facility(**over.pop("terms", {}))
    df = frame(over.pop("rows", BOOK))
    receipt = derive_eligibility(df, fac)
    return evaluate(df, facility=fac, client_id=fac.client_id,
                    eligibility_derivation=receipt,
                    as_of_date="2025-11-30", run_id="mi_2025_11",
                    portfolio_scope={"client_id": fac.client_id},
                    concentration_results=CONCENTRATION,
                    concentration_rule_version="3",
                    concentration_library_version="1.0.0",
                    **over)


# --------------------------------------------------------------------------- #
# The measure registry
# --------------------------------------------------------------------------- #
class TestTheMeasureRegistry:
    def test_every_target_measure_for_a_later_MI_sprint_exists(self):
        # The names a later sprint would register against. Renaming one is a
        # breaking change, which is why they are asserted here by name.
        for measure_id in ("eligible_balance", "ineligible_balance",
                           "borrowing_base", "facility_commitment",
                           "facility_drawn", "borrowing_base_headroom",
                           "borrowing_base_utilisation", "facility_utilisation",
                           "borrowing_base_deficiency",
                           "nearest_concentration_limit",
                           "nearest_concentration_headroom"):
            assert measure_id in MEASURE_IDS

    def test_every_measure_declares_a_unit_and_a_definition(self):
        for definition in measure_definitions():
            assert definition["unit"] in ("currency", "percent", "count", "label")
            assert definition["definition"].strip()

    def test_the_registry_order_is_stable(self):
        assert MEASURE_IDS == tuple(d["measure_id"] for d in measure_definitions())

    def test_an_unknown_measure_raises_rather_than_returning_nothing(self):
        with pytest.raises(KeyError):
            snapshot().measure("profit")


class TestTheMeasures:
    def test_the_calculated_measures_carry_the_hand_checked_figures(self):
        s = snapshot()
        assert s.measure("eligible_balance") == 100_000_000.0
        assert s.measure("gross_borrowing_base") == 103_000_000.0
        assert s.measure("borrowing_base") == 103_000_000.0
        assert s.measure("facility_commitment") == 250_000_000.0
        assert s.measure("advance_rate") == 103.0
        assert s.measure("concentration_limit_denominator") == 100_000_000.0

    def test_a_measure_with_no_input_is_NOT_CALCULABLE_not_zero(self):
        s = snapshot()
        for measure_id in ("facility_drawn", "borrowing_base_headroom",
                           "borrowing_base_deficiency",
                           "borrowing_base_utilisation", "facility_utilisation"):
            assert s.measure(measure_id) == NOT_CALCULABLE

    def test_the_binding_concentration_is_deterministic(self):
        s = snapshot()
        assert s.measure("nearest_concentration_limit") == "East of England"
        assert s.measure("breached_concentration_count") == 1

    def test_a_drawn_facility_produces_every_measure(self):
        s = snapshot(terms={"current_drawn_amount": 80_000_000.0})
        assert s.measure("facility_drawn") == 80_000_000.0
        assert s.measure("borrowing_base_headroom") == 23_000_000.0
        assert s.measure("borrowing_base_deficiency") == 0.0
        assert s.measure("facility_utilisation") == 32.0

    def test_no_configured_facility_yields_an_explicit_empty_state(self):
        df = frame(BOOK)
        s = evaluate(df, client_id="a_client_with_no_warehouse")
        assert s.available is False
        assert "No funding facility is configured" in s.reason
        # Every measure is NOT_CALCULABLE — not zero, and not absent.
        assert set(s.measures().values()) == {NOT_CALCULABLE}
        assert s.to_dict()["measures"] == {}


# --------------------------------------------------------------------------- #
# Independence from the dashboard
# --------------------------------------------------------------------------- #
class TestTheEngineDoesNotDependOnTheDashboard:
    def test_the_calculation_runs_in_a_process_with_no_UI_import(self):
        """A subprocess that BANS the UI and web frameworks outright.

        Importing them is made an ImportError, so the borrowing base cannot
        quietly acquire a dashboard dependency without this failing.
        """
        program = """
import sys, builtins
BANNED = ("streamlit", "fastapi", "starlette", "uvicorn",
          "mi_agent_api.app", "mi_agent_api.concentration_tests_api")
real_import = builtins.__import__
def guarded(name, *a, **k):
    if any(name == b or name.startswith(b + ".") for b in BANNED):
        raise ImportError(f"{name} must not be reachable from the engine")
    return real_import(name, *a, **k)
builtins.__import__ = guarded

import pandas as pd
from mi_agent.borrowing_base.eligibility import derive_eligibility
from mi_agent.borrowing_base.config import load_facility
from mi_agent.borrowing_base.service import evaluate

facility = load_facility("ere_funding_uk")
df = pd.DataFrame({"loan_id": ["L0"],
                   "current_outstanding_balance": [100_000_000.0]})
derive_eligibility(df, facility)
snapshot = evaluate(df, facility=facility, client_id="ere_funding_uk")
assert snapshot.measure("borrowing_base") == 103_000_000.0
assert snapshot.measure("borrowing_base_headroom") == "NOT_CALCULABLE"
for banned in BANNED:
    assert banned not in sys.modules, banned
print("OK")
"""
        result = subprocess.run([sys.executable, "-c", program],
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_no_engine_module_imports_a_UI_or_web_framework(self):
        import pathlib
        import re
        package = pathlib.Path("mi_agent/borrowing_base")
        banned = re.compile(
            r"^\s*(?:import|from)\s+(streamlit|fastapi|starlette|uvicorn"
            r"|mi_agent_api)\b", re.MULTILINE)
        for module in package.glob("*.py"):
            assert not banned.search(module.read_text(encoding="utf-8")), module


# --------------------------------------------------------------------------- #
# The receipt
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def receipt():
    return snapshot(terms={"current_drawn_amount": 120_000_000.0}).receipt


class TestTheReceipt:
    @pytest.mark.parametrize("key", [
        "facility_id", "facility_config_version", "as_of_date",
        "portfolio_scope", "financing_portfolio_balance",
        "eligibility_rule_version", "eligible_balance", "ineligible_balance",
        "undetermined_balance", "concentration_denominator", "advance_rate",
        "facility_commitment", "gross_borrowing_base",
        "available_borrowing_base", "current_drawn_amount", "headroom",
        "deficiency", "concentration_rule_version", "concentration_results",
        "prototype_assumptions_used", "timestamp",
    ])
    def test_the_required_field_is_present(self, receipt, key):
        assert key in receipt

    def test_it_records_the_figures_it_reported(self, receipt):
        assert receipt["eligible_balance"] == 100_000_000.0
        assert receipt["gross_borrowing_base"] == 103_000_000.0
        assert receipt["available_borrowing_base"] == 103_000_000.0
        assert receipt["current_drawn_amount"] == 120_000_000.0
        # The receipt keeps the NEGATIVE headroom, whatever the UI shows.
        assert receipt["headroom"] == -17_000_000.0
        assert receipt["deficiency"] == 17_000_000.0

    def test_it_states_the_equations_it_applied(self, receipt):
        assert "gross_borrowing_base = eligible_balance x advance_rate" in \
            receipt["equations"]
        assert "deficiency = ABS(MIN(headroom, 0))" in receipt["equations"]

    def test_it_hashes_the_exact_terms_it_calculated_on(self, receipt):
        assert len(receipt["facility_config_hash"]) == 16
        assert receipt["facility_environment"] == "prototype"

    def test_a_LOADED_facility_records_where_its_terms_came_from(self):
        from mi_agent.borrowing_base.config import load_facility
        fac = load_facility("ere_funding_uk")
        df = frame(BOOK)
        derivation = derive_eligibility(df, fac)
        receipt = evaluate(df, facility=fac, client_id="ere_funding_uk",
                           eligibility_derivation=derivation).receipt
        assert receipt["facility_config_source"].startswith("platform_register:")
        assert receipt["facility_config_version"]

    def test_it_carries_every_concentration_result_with_its_source_wording(
            self, receipt):
        rows = {r["display_name"]: r for r in receipt["concentration_results"]}
        assert set(rows) == {"East of England", "Scotland"}
        assert rows["East of England"]["source_text"] == "UKH East of England [25]%"
        assert rows["East of England"]["denominator_value"] == 100_000_000.0

    def test_it_flags_every_prototype_assumption_that_was_in_play(self, receipt):
        assert any("PROTOTYPE ASSUMPTION" in note
                   for note in receipt["prototype_assumptions_used"])

    def test_it_lists_what_could_not_be_calculated_when_something_cannot(self):
        receipt = snapshot().receipt
        assert set(receipt["not_calculable_measures"]) == {
            "current_drawn_amount", "headroom", "deficiency",
            "borrowing_base_utilisation_pct", "facility_utilisation_pct"}
        assert receipt["missing_inputs"] == ["current_drawn_amount"]

    def test_it_shows_the_reconciliation_invariants_holding(self, receipt):
        assert receipt["reconciles"] is True
        assert all(i["holds"] for i in receipt["invariants"])

    def test_the_same_inputs_produce_the_same_content_hash(self):
        first = snapshot().receipt
        second = snapshot().receipt
        assert first["content_hash"] == second["content_hash"]

    def test_a_different_advance_rate_produces_a_different_hash(self):
        first = snapshot().receipt
        other = snapshot(terms={"advance_rate": 0.9}).receipt
        assert first["content_hash"] != other["content_hash"]

    def test_a_production_book_with_no_rules_records_it_as_undetermined(self):
        fac = production_facility()
        df = frame(BOOK)
        derivation = derive_eligibility(df, fac)
        receipt = evaluate(df, facility=fac, client_id=fac.client_id,
                           eligibility_derivation=derivation).receipt
        assert receipt["eligible_balance"] == 0.0
        assert receipt["undetermined_balance"] == 100_000_000.0
        assert receipt["eligibility_governed"] is False
        assert receipt["prototype_assumptions_used"] == []


class TestTheMIQueryAgentReachesTheEngineThroughOneOwner:
    """The MI sprint has happened. What the old guard protected still holds.

    The earlier version of this class asserted that no MI module referenced
    the borrowing base at all, "until the MI sprint happens — they will delete
    these and mean it". This is that sprint, and the property worth keeping is
    not silence but SINGULARITY: the MI Query Agent reaches the engine through
    one registered recogniser, one integration module and the same API seam
    the dashboard uses. No parsing or routing module imports the calculator or
    the eligibility derivation.
    """

    CALCULATION_OWNERS = ("borrowing_base.calculator", "borrowing_base.eligibility",
                          "borrowing_base import calculator",
                          "borrowing_base import eligibility")

    NEVER_CALCULATE = (
        "mi_agent/llm_query_parser.py",
        "mi_agent/mi_query_spec.py",
        "mi_agent/mi_query_executor.py",
        "mi_agent/mi_query_validator.py",
        "mi_agent/mi_query_contract.py",
        "mi_agent/query_plan.py",
        "mi_agent/query_plan_compiler.py",
        "mi_agent/business_semantics.py",
        "mi_agent/semantic_resolver.py",
        "mi_agent_api/chat_routing.py",
        "mi_agent_api/analytical_plan.py",
        "mi_agent_api/borrowing_base_query.py",
        "mi_agent_api/temporal_query.py",
        "mi_agent_api/evolution.py",
    )

    @pytest.mark.parametrize("path", NEVER_CALCULATE)
    def test_no_MI_parsing_or_routing_module_imports_the_calculation_owner(self, path):
        import pathlib
        module = pathlib.Path(path)
        if not module.exists():          # a module renamed upstream
            pytest.skip(f"{path} is not present in this tree")
        text = module.read_text(encoding="utf-8")
        for owner in self.CALCULATION_OWNERS:
            assert owner not in text, (
                f"{path} reaches {owner} directly. The MI Query Agent must "
                "consume the governed borrowingBase envelope through "
                "borrowing_base_api, never calculate for itself.")

    def test_the_MI_route_is_registered_through_the_recogniser_registry(self):
        from mi_agent_api.chat_routing import REGISTRY
        assert "borrowing_base" in REGISTRY
        # ONE registration, from the integration module, not a hand-ordered branch.
        import pathlib
        routing = pathlib.Path("mi_agent_api/chat_routing.py").read_text(encoding="utf-8")
        assert routing.count("_borrowing_base.recogniser()") == 1

    def test_the_parser_carries_only_the_owner_nouns_not_any_calculation(self):
        import pathlib
        parser = pathlib.Path("mi_agent/llm_query_parser.py").read_text(encoding="utf-8")
        assert "_BORROWING_BASE_NOUNS" in parser
        assert "advance_rate" not in parser and "borrowingBase" not in parser

    def test_the_measures_are_available_to_register_when_that_sprint_happens(self):
        # The seam itself: named, unit-carrying, definition-carrying measures
        # that a registration step can consume without touching this package.
        definitions = {d["measure_id"]: d for d in measure_definitions()}
        assert definitions["borrowing_base"]["unit"] == "currency"
        assert definitions["facility_utilisation"]["unit"] == "percent"
        assert definitions["nearest_concentration_limit"]["unit"] == "label"
