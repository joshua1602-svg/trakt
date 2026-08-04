#!/usr/bin/env python3
"""tests/test_simulation_framework.py — unit tests for the hardening framework.

Fast, dependency-light checks of the framework's OWN contracts: the manifest
catalogue, deterministic seeding, loan-state transitions, balance
reconciliation, dialect formatting, the independence of the reference truth, and
the failure classification. Nothing here runs the production pipeline; the
end-to-end coverage lives in ``tests/test_simulation_pipeline.py``.

Run: python -m pytest tests/test_simulation_framework.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulation import SIMULATION_VERSION  # noqa: E402
from simulation import manifests as M  # noqa: E402
from simulation._registry import unsupported_fields  # noqa: E402
from simulation.assets import all_profiles, get_profile  # noqa: E402
from simulation.assets._common import REPORTED_STATUS  # noqa: E402
from simulation.dialects import registered, render  # noqa: E402
from simulation.dialects import _format as F  # noqa: E402
from simulation.dialects.vocabulary import all_aliases, headers_for  # noqa: E402
from simulation.history import canonical_frame_rows, generate_history  # noqa: E402
from simulation.knobs import INTENT_MODIFIERS, UnknownIntentError, resolve_knobs  # noqa: E402
from simulation.models import (  # noqa: E402
    ASSET_CLASSES,
    CLOSED_STATUSES,
    DIALECTS,
    EXPECT_FAILURE,
    OPEN_STATUSES,
    POSITIVE_MOVEMENTS,
    CaseManifest,
    FailureClass,
    SchemaEvolution,
)
from simulation.reference_truth import build_expected_truth  # noqa: E402


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
class TestCatalogue(unittest.TestCase):
    def test_catalogue_validates(self):
        self.assertEqual(M.validate_catalogue(), {})

    def test_six_economic_cases_per_asset_family(self):
        """The definition of done: six meaningful cases per family."""
        economic = [c for c in M.all_cases()
                    if c.expectation != EXPECT_FAILURE]
        for asset_class in ASSET_CLASSES:
            cases = [c for c in economic if c.asset_class == asset_class]
            self.assertEqual(len(cases), 6, f"{asset_class}: {[c.case_id for c in cases]}")

    def test_every_case_has_at_least_six_monthly_snapshots(self):
        for case in M.all_cases():
            self.assertGreaterEqual(case.months, 6, case.case_id)
            self.assertEqual(len(case.reporting_dates()), case.months)

    def test_every_economic_case_renders_three_dialects(self):
        for case in M.all_cases():
            if case.expectation == EXPECT_FAILURE:
                continue  # a guard case deliberately renders one dialect
            self.assertEqual(len(case.dialects), 3, case.case_id)
            self.assertEqual(set(case.dialects), set(DIALECTS), case.case_id)

    def test_every_case_states_a_hardening_purpose(self):
        for case in M.all_cases():
            self.assertTrue(case.purpose.strip(), case.case_id)
            self.assertGreater(len(case.purpose), 40, case.case_id)

    def test_seeds_are_unique(self):
        seeds = [c.seed for c in M.all_cases()]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_declared_schema_evolution_is_covered(self):
        """The catalogue must exercise the declared drift vocabulary."""
        kinds = {e.kind for c in M.all_cases() for e in c.schema_evolution}
        for expected in ("reorder_columns", "add_optional_column",
                         "drop_optional_column", "change_alias",
                         "change_enum_synonyms", "switch_format",
                         "restate_prior", "ambiguous_schema",
                         "drop_mandatory_column"):
            self.assertIn(expected, kinds)

    def test_profiles_resolve(self):
        smoke = M.cases_for_profile(M.PROFILE_SMOKE)
        self.assertEqual({c.asset_class for c in smoke}, set(ASSET_CLASSES))
        self.assertEqual(len(M.cases_for_profile(M.PROFILE_STANDARD)),
                         len(M.all_cases()))
        self.assertTrue(M.cases_for_profile(M.PROFILE_PERFORMANCE))

    def test_manifest_round_trips(self):
        for case in M.all_cases():
            self.assertEqual(CaseManifest.from_dict(case.to_dict()), case)

    def test_manifest_persists_and_reloads(self):
        case = M.get_case("bridge_maturity_wall_v1")
        with tempfile.TemporaryDirectory() as td:
            path = M.write_manifest(case, Path(td) / "manifest.json")
            self.assertEqual(M.read_manifest(path), case)

    def test_run_key_identifies_case_seed_asset_and_version(self):
        case = M.get_case("bridge_maturity_wall_v1")
        self.assertIn(case.case_id, case.run_key)
        self.assertIn(str(case.seed), case.run_key)
        self.assertIn(case.asset_class, case.run_key)
        self.assertIn(f"v{SIMULATION_VERSION}", case.run_key)


class TestManifestValidation(unittest.TestCase):
    def _case(self, **kw) -> CaseManifest:
        from dataclasses import replace
        return replace(M.get_case("bridge_clean_short_duration_v1"), **kw)

    def test_a_short_history_is_rejected(self):
        problems = M.validate_manifest(self._case(months=3))
        self.assertTrue(any("six monthly snapshots" in p for p in problems))

    def test_a_purposeless_case_is_rejected(self):
        problems = M.validate_manifest(self._case(purpose="  "))
        self.assertTrue(any("purpose is mandatory" in p for p in problems))

    def test_an_unknown_intent_is_rejected(self):
        problems = M.validate_manifest(self._case(economic_intent=("nonsense",)))
        self.assertTrue(any("no declared modifier" in p for p in problems))

    def test_a_case_may_not_expect_a_production_defect(self):
        problems = M.validate_manifest(self._case(
            expectation=EXPECT_FAILURE,
            expected_failure_class=FailureClass.PRODUCTION_DEFECT))
        self.assertTrue(any("not declarable" in p for p in problems))

    def test_an_unknown_schema_evolution_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            SchemaEvolution(from_period=0, kind="freestyle", purpose="x")

    def test_a_non_green_case_must_name_its_limits(self):
        from simulation.models import RISK_RED, RiskIntent
        problems = M.validate_manifest(
            self._case(risk_intent=RiskIntent(RISK_RED, ())))
        self.assertTrue(any("targeted limits" in p for p in problems))


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
class TestDeterminism(unittest.TestCase):
    CASE = "bridge_extension_heavy_v1"

    def test_same_seed_reproduces_the_same_history(self):
        case = M.get_case(self.CASE)
        first = build_expected_truth(case, generate_history(case))
        second = build_expected_truth(case, generate_history(case))
        self.assertEqual(json.dumps(first, sort_keys=True, default=str),
                         json.dumps(second, sort_keys=True, default=str))

    def test_a_different_seed_produces_a_different_history(self):
        from dataclasses import replace
        case = M.get_case(self.CASE)
        base = build_expected_truth(case, generate_history(case))
        other = build_expected_truth(replace(case, seed=case.seed + 1),
                                     generate_history(replace(case, seed=case.seed + 1)))
        self.assertNotEqual(base["periods"][-1]["closing_balance"],
                            other["periods"][-1]["closing_balance"])

    def test_generation_is_independent_of_module_level_randomness(self):
        """A global seed must not change a case: the generator owns its stream."""
        import random

        import numpy as np

        case = M.get_case(self.CASE)
        random.seed(1); np.random.seed(1)
        first = generate_history(case).periods[-1].closing_balance
        random.seed(999); np.random.seed(999)
        second = generate_history(case).periods[-1].closing_balance
        self.assertEqual(first, second)


# --------------------------------------------------------------------------- #
# Loan state + reconciliation
# --------------------------------------------------------------------------- #
class TestLoanStateAndReconciliation(unittest.TestCase):
    def test_every_case_reconciles_every_period(self):
        for case in M.all_cases():
            history = generate_history(case)
            for period in history.periods:
                self.assertTrue(
                    period.reconciles(),
                    f"{case.case_id} {period.reporting_date}: "
                    f"{period.movement_totals}")

    def test_opening_equals_prior_closing(self):
        history = generate_history(M.get_case("asset_finance_balloon_residual_v1"))
        for prior, current in zip(history.periods, history.periods[1:]):
            self.assertAlmostEqual(prior.closing_balance,
                                   current.opening_balance, places=2)

    def test_loans_on_book_are_open_and_exits_are_closed(self):
        history = generate_history(M.get_case("bridge_arrears_default_enforcement_v1"))
        for period in history.periods:
            for loan in period.loans:
                self.assertIn(loan.status, OPEN_STATUSES)
                self.assertGreaterEqual(loan.closing_balance, 0.0)
            for loan in period.exits:
                self.assertIn(loan.status, CLOSED_STATUSES)
                self.assertEqual(loan.closing_balance, 0.0)
                self.assertTrue(loan.exit_date)

    def test_the_impairment_ladder_never_skips_a_rung(self):
        """performing → arrears → default → enforcement, with cure only from arrears."""
        from simulation.models import (STATUS_ARREARS, STATUS_DEFAULT,
                                       STATUS_ENFORCEMENT, STATUS_PERFORMING)
        allowed = {
            STATUS_PERFORMING: {STATUS_PERFORMING, STATUS_ARREARS},
            STATUS_ARREARS: {STATUS_ARREARS, STATUS_PERFORMING, STATUS_DEFAULT},
            STATUS_DEFAULT: {STATUS_DEFAULT, STATUS_ENFORCEMENT},
            STATUS_ENFORCEMENT: {STATUS_ENFORCEMENT},
        }
        history = generate_history(M.get_case("bridge_arrears_default_enforcement_v1"))
        previous = {}
        for period in history.periods:
            for loan in period.loans:
                was = previous.get(loan.loan_id)
                if was is not None:
                    self.assertIn(loan.status, allowed[was],
                                  f"{loan.loan_id}: {was} -> {loan.status}")
                previous[loan.loan_id] = loan.status

    def test_a_write_off_splits_the_balance_into_proceeds_and_loss(self):
        history = generate_history(
            M.get_case("asset_finance_delinquency_repossession_v1"))
        seen = False
        for period in history.periods:
            for loan in period.exits:
                if loan.exit_reason != "repossession_and_disposal":
                    continue
                seen = True
                total = sum(loan.movements.get(k, 0.0)
                            for k in ("write_off", "asset_disposal"))
                self.assertAlmostEqual(total, loan.opening_balance
                                       + loan.movements.get("rolled_up_interest", 0.0)
                                       - loan.movements.get("principal_amortisation", 0.0),
                                       places=1)
        self.assertTrue(seen, "no repossession occurred in the repossession case")

    def test_loan_identifiers_are_stable_across_the_history(self):
        history = generate_history(M.get_case("equity_release_seasoned_rollup_v1"))
        origination = {}
        for period in history.periods:
            for loan in period.loans:
                if loan.loan_id in origination:
                    self.assertEqual(origination[loan.loan_id], loan.origination_date)
                origination[loan.loan_id] = loan.origination_date

    def test_a_restatement_case_records_a_signed_restatement(self):
        case = M.get_case("equity_release_redemption_heavy_v1")
        truth = build_expected_truth(case, generate_history(case))
        restated = [p for p in truth["periods"] if p["restatement"]]
        self.assertTrue(restated)
        self.assertLess(restated[0]["restatement"], 0.0)

    def test_no_pipeline_record_is_ever_generated(self):
        """Funded books only — the framework's scope boundary, enforced."""
        forbidden = {"pipeline_stage", "opportunity_id", "application_id",
                     "forecast_funded_balance", "forecast_funding_probability"}
        for case in M.all_cases():
            profile = get_profile(case.asset_class)
            self.assertEqual(set(profile.canonical_fields) & forbidden, set(),
                             case.case_id)


class TestKnobs(unittest.TestCase):
    def test_every_intent_has_a_declared_modifier(self):
        for case in M.all_cases():
            for intent in case.economic_intent:
                self.assertIn(intent, INTENT_MODIFIERS, case.case_id)

    def test_an_unknown_intent_raises(self):
        with self.assertRaises(UnknownIntentError):
            resolve_knobs("bridge", ("not_a_real_intent",))

    def test_intents_apply_in_declaration_order(self):
        first = resolve_knobs("bridge", ("clean_performing",))
        second = resolve_knobs("bridge", ("clean_performing", "arrears_transitions"))
        self.assertEqual(first.arrears_entry_rate, 0.0)
        self.assertGreater(second.arrears_entry_rate, 0.0)


# --------------------------------------------------------------------------- #
# Asset profiles
# --------------------------------------------------------------------------- #
class TestAssetProfiles(unittest.TestCase):
    def test_every_profile_emits_only_selectable_canonical_fields(self):
        """No asset class may rely on a field production would discard."""
        for asset_class, profile in all_profiles().items():
            self.assertEqual(unsupported_fields(asset_class, profile.canonical_fields),
                             [], asset_class)

    def test_equipment_finance_is_an_asset_finance_subtype(self):
        from simulation.models import ASSET_SUBTYPES
        self.assertIn("equipment_finance", ASSET_SUBTYPES["asset_finance"])
        self.assertNotIn("equipment_finance", ASSET_CLASSES)

    def test_reported_statuses_satisfy_the_production_business_rules(self):
        """Reported tokens must be the ones ARR001 / DEF001 / BAL103 accept."""
        self.assertEqual(REPORTED_STATUS["arrears"], "ARRE")
        self.assertEqual(REPORTED_STATUS["default"], "DEFAULT")
        self.assertEqual(REPORTED_STATUS["enforcement"], "DEFAULT")
        self.assertIn(REPORTED_STATUS["redeemed"], ("REPAID", "REDEEMED", "CLOSED"))

    def test_every_risk_test_names_a_governed_library_metric(self):
        from mi_agent.concentration_tests.library import load_library
        library = load_library()
        for case in M.all_cases():
            for test in get_profile(case.asset_class).risk_tests(case):
                metric = library.get(test["metric_id"])
                self.assertIsNotNone(metric, f"{case.case_id}: {test['metric_id']}")
                self.assertEqual(metric.implementation_status, "implemented")

    def test_a_stressed_case_lowers_only_its_targeted_thresholds(self):
        from dataclasses import replace

        from simulation.models import RISK_GREEN, RiskIntent
        case = M.get_case("bridge_maturity_wall_v1")
        stressed = {t["test_id"]: t["threshold"]
                    for t in get_profile(case.asset_class).risk_tests(case)}
        relaxed = {t["test_id"]: t["threshold"]
                   for t in get_profile(case.asset_class).risk_tests(
                       replace(case, risk_intent=RiskIntent(RISK_GREEN, ())))}
        moved = {k for k in stressed if stressed[k] != relaxed[k]}
        self.assertEqual(moved, {"br_maturity_90_days"})


# --------------------------------------------------------------------------- #
# Dialects
# --------------------------------------------------------------------------- #
class TestDialects(unittest.TestCase):
    def test_all_three_dialects_are_registered(self):
        self.assertEqual(set(registered()), set(DIALECTS))

    def test_every_emitted_header_is_covered_by_the_approved_contract(self):
        """A dialect may not rely on a header no onboarding contract approves."""
        import yaml

        for asset_class in ASSET_CLASSES:
            contract_path = (_REPO / "simulation" / "aliases" / asset_class
                             / "aliases_client_contract.yaml")
            contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            approved = {alias for entry in contract.values()
                        for alias in (entry.get("aliases") or [])}
            for flavour in ("lender", "locale"):
                for field, header in headers_for(asset_class, flavour).items():
                    self.assertIn(header, approved,
                                  f"{asset_class}/{flavour}: {field} -> {header}")

    def test_the_committed_contract_matches_the_vocabulary(self):
        """A regenerated contract must equal the committed one."""
        import yaml

        from simulation.tools.build_alias_contracts import contract_for
        for asset_class in ASSET_CLASSES:
            path = (_REPO / "simulation" / "aliases" / asset_class
                    / "aliases_client_contract.yaml")
            committed = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(committed, contract_for(asset_class), asset_class)

    def test_the_clean_dialect_uses_canonical_headers(self):
        for asset_class in ASSET_CLASSES:
            headers = headers_for(asset_class, "canonical")
            self.assertEqual(set(headers), set(headers.values()))

    def test_the_three_vocabularies_are_materially_different(self):
        for asset_class in ASSET_CLASSES:
            canonical = set(headers_for(asset_class, "canonical").values())
            lender = set(headers_for(asset_class, "lender").values())
            locale = set(headers_for(asset_class, "locale").values())
            self.assertEqual(canonical & lender, set(), asset_class)
            self.assertEqual(canonical & locale, set(), asset_class)
            self.assertEqual(lender & locale, set(), asset_class)

    def test_each_alias_resolves_to_exactly_one_canonical_field(self):
        for asset_class in ASSET_CLASSES:
            seen = {}
            for field, aliases in all_aliases(asset_class).items():
                for alias in aliases:
                    self.assertNotIn(alias, seen,
                                     f"{asset_class}: {alias} -> "
                                     f"{seen.get(alias)} and {field}")
                    seen[alias] = field

    def test_dialects_render_and_differ(self):
        case = M.get_case("asset_finance_mixed_collateral_v1")
        history = generate_history(case)
        rows = canonical_frame_rows(case, history, 0)
        with tempfile.TemporaryDirectory() as td:
            paths = {d: render(d, rows, case, period_index=0,
                               reporting_date=history.periods[0].reporting_date,
                               out_dir=Path(td) / d,
                               evolution=case.schema_evolution)
                     for d in case.dialects}
            self.assertEqual(len(paths), 3)
            for path in paths.values():
                self.assertGreater(path.stat().st_size, 1000)
            sizes = {p.read_bytes() for p in paths.values()}
            self.assertEqual(len(sizes), 3, "dialects produced identical bytes")

    def test_a_declared_reorder_changes_the_column_order(self):
        case = M.get_case("equity_release_seasoned_rollup_v1")
        history = generate_history(case)
        with tempfile.TemporaryDirectory() as td:
            first = render("clean_csv", canonical_frame_rows(case, history, 0),
                           case, period_index=0, reporting_date="2025-01-31",
                           out_dir=Path(td) / "p0",
                           evolution=case.schema_evolution)
            later = render("clean_csv", canonical_frame_rows(case, history, 3),
                           case, period_index=3, reporting_date="2025-04-30",
                           out_dir=Path(td) / "p3",
                           evolution=case.schema_evolution)
            head_first = first.read_text(encoding="utf-8").splitlines()[0]
            head_later = later.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotEqual(head_first, head_later)
            self.assertEqual(sorted(head_first.split(",")),
                             sorted(head_later.split(",")))

    def test_an_ambiguous_schema_emits_unresolvable_headers(self):
        case = M.get_case("bridge_unsupported_schema_guard_v1")
        history = generate_history(case)
        with tempfile.TemporaryDirectory() as td:
            path = render("clean_csv", canonical_frame_rows(case, history, 0),
                          case, period_index=0, reporting_date="2025-01-31",
                          out_dir=Path(td), evolution=case.schema_evolution)
            headers = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertTrue(all(h.startswith("col_") for h in headers), headers)

    def test_a_withheld_mandatory_column_is_absent(self):
        case = M.get_case("equity_release_missing_mandatory_guard_v1")
        history = generate_history(case)
        with tempfile.TemporaryDirectory() as td:
            path = render("clean_csv", canonical_frame_rows(case, history, 0),
                          case, period_index=0, reporting_date="2025-01-31",
                          out_dir=Path(td), evolution=case.schema_evolution)
            headers = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertNotIn("current_outstanding_balance", headers)
            self.assertNotIn("current_principal_balance", headers)


class TestFormatters(unittest.TestCase):
    def test_date_presentations(self):
        self.assertEqual(F.iso_date("2025-01-31"), "2025-01-31")
        self.assertEqual(F.day_first_date("2025-01-31"), "31/01/2025")
        self.assertEqual(F.long_date("2025-01-31"), "31 Jan 2025")

    def test_numeric_presentations(self):
        self.assertEqual(F.decimal_point(1234567.891), "1234567.89")
        self.assertEqual(F.thousands_grouped(1234567.891), "1,234,567.89")
        self.assertEqual(F.percent_string(43.105), "43.10%")
        self.assertEqual(F.percent_fraction(43.1, 4), "0.4310")

    def test_boolean_styles(self):
        self.assertEqual(F.boolean("Y", "yn"), "Y")
        self.assertEqual(F.boolean("Y", "yesno"), "Yes")
        self.assertEqual(F.boolean("N", "binary"), "0")

    def test_region_codes_cover_every_region(self):
        from simulation.geo import REGIONS
        for region in REGIONS:
            self.assertIn(region, F.REGION_CODES)


# --------------------------------------------------------------------------- #
# Reference truth
# --------------------------------------------------------------------------- #
class TestReferenceTruthIndependence(unittest.TestCase):
    """The expected truth must never be computed by the code it checks."""

    @staticmethod
    def _imported_modules(path: Path):
        """Every module name the file imports, from the parsed AST.

        Parsed rather than grepped so a prose mention of ``mi_agent`` in the
        module docstring is not mistaken for a dependency on it.
        """
        import ast

        modules = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
        return modules

    def test_reference_truth_imports_no_production_analytics(self):
        modules = self._imported_modules(
            _REPO / "simulation" / "reference_truth.py")
        for module in modules:
            root = module.split(".")[0]
            self.assertNotIn(root, ("mi_agent", "mi_agent_api", "analytics",
                                    "analytics_lib", "engine", "trakt_core",
                                    "snapshot", "pandas", "numpy"),
                             f"reference_truth must not import {module}")

    def test_reference_truth_uses_no_pandas(self):
        self.assertNotIn(
            "pandas", {m.split(".")[0] for m in self._imported_modules(
                _REPO / "simulation" / "reference_truth.py")})

    def test_expected_truth_covers_the_required_figures(self):
        case = M.get_case("bridge_maturity_wall_v1")
        truth = build_expected_truth(case, generate_history(case))
        period = truth["periods"][-1]
        for key in ("loan_count", "opening_balance", "closing_balance",
                    "funded_additions", "redemptions", "write_offs",
                    "recoveries", "period_movement",
                    "weighted_average_current_ltv", "largest_concentrations",
                    "status_balances", "status_counts", "movement_reconciles",
                    "vintage_balances", "cohort_balances", "asset_measures"):
            self.assertIn(key, period)
        for key in ("expected_risk_status", "expected_regime",
                    "expected_validation_warnings", "expected_outcome"):
            self.assertIn(key, truth)

    def test_movement_delta_reconciles_by_hand(self):
        case = M.get_case("asset_finance_delinquency_repossession_v1")
        truth = build_expected_truth(case, generate_history(case))
        for period in truth["periods"]:
            self.assertTrue(period["movement_reconciles"])
            self.assertAlmostEqual(
                period["opening_balance"] + period["movement_delta"],
                period["closing_balance"], places=1)

    def test_positive_movements_are_the_additions(self):
        self.assertEqual(
            set(POSITIVE_MOVEMENTS),
            {"funded_addition", "acquired_addition", "rolled_up_interest",
             "further_advance", "restatement"})


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #
class TestFailureClassification(unittest.TestCase):
    def test_a_case_may_never_declare_a_defect_as_its_expected_outcome(self):
        for forbidden in (FailureClass.PRODUCTION_DEFECT,
                          FailureClass.SIMULATION_DEFECT,
                          FailureClass.UNCLASSIFIED_EXCEPTION,
                          FailureClass.MI_ASSERTION_FAILURE,
                          FailureClass.AGENT_GROUNDING_FAILURE,
                          FailureClass.REGULATORY_ARTEFACT_FAILURE):
            self.assertNotIn(forbidden, FailureClass.DECLARABLE)

    def test_declared_guard_cases_use_declarable_classes(self):
        for case in M.all_cases():
            if case.expectation != EXPECT_FAILURE:
                continue
            self.assertIn(case.expected_failure_class, FailureClass.DECLARABLE)

    def test_a_case_result_is_ok_only_when_it_ended_as_declared(self):
        from simulation.models import CaseResult, StageResult
        result = CaseResult(case_id="x", seed=1, asset_class="bridge",
                            simulation_version=1, expectation=EXPECT_FAILURE,
                            expected_failure_class=FailureClass.MISSING_MANDATORY_DATA)
        self.assertFalse(result.ok, "a failure case that did not fail is not ok")
        result.stages.append(StageResult(
            stage="ingest", ok=False,
            failure_class=FailureClass.MISSING_MANDATORY_DATA))
        self.assertTrue(result.ok)
        result.stages[-1].failure_class = FailureClass.PRODUCTION_DEFECT
        self.assertFalse(result.ok, "the wrong failure class is not a pass")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
