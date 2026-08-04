#!/usr/bin/env python3
"""tests/test_simulation_multi_source.py — one SPV, two funded deliveries.

Covers the multi-source case that exercises ``engine/platform_assembler.py``
through the production Assembler Agent, and the reporting rule that came out of
finding the assembler had never actually run: a run summary must list the
production modules that were OBSERVED to execute, never a declared list.

Run: python -m pytest tests/test_simulation_multi_source.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulation import manifests as M  # noqa: E402
from simulation import pipeline as P  # noqa: E402
from simulation import runner as R  # noqa: E402
from simulation.history import generate_history  # noqa: E402
from simulation.models import (  # noqa: E402
    SOURCE_ACQUIRED,
    SOURCE_DIRECT,
    STAGE_ASSEMBLE,
    FundedSource,
)
from simulation.reference_truth import build_expected_truth, combine_truths  # noqa: E402

CASE_ID = "equity_release_spv_multi_source_v1"


class TestMultiSourceManifest(unittest.TestCase):
    def setUp(self):
        self.case = M.get_case(CASE_ID)

    def test_the_case_is_one_client_and_one_spv(self):
        self.assertEqual(len({s.portfolio_id for s in self.case.sources}), 2)
        # Direct/acquired is provenance INSIDE one SPV, not a second SPV.
        self.assertTrue(self.case.spv_id)
        self.assertEqual(self.case.client_id, "sim_multi_source")

    def test_both_provenance_types_are_carried(self):
        types = {s.source_portfolio_type for s in self.case.sources}
        self.assertEqual(types, {SOURCE_DIRECT, SOURCE_ACQUIRED})

    def test_the_acquired_source_declares_its_acquisition_date(self):
        acquired = next(s for s in self.case.sources
                        if s.source_portfolio_type == SOURCE_ACQUIRED)
        self.assertTrue(acquired.acquisition_date)
        self.assertTrue(acquired.seller_name)

    def test_reporting_dates_are_aligned_by_construction(self):
        for source in self.case.sources:
            self.assertEqual(self.case.source_case(source).reporting_dates(),
                             self.case.reporting_dates())

    def test_the_sources_draw_distinct_seed_streams(self):
        offsets = [s.seed_offset for s in self.case.sources]
        self.assertEqual(len(set(offsets)), len(offsets))

    def test_shares_sum_to_one(self):
        self.assertEqual(round(sum(s.share for s in self.case.sources), 9), 1.0)

    def test_the_catalogue_accepts_it(self):
        self.assertEqual(M.validate_manifest(self.case), [])

    def test_it_round_trips(self):
        self.assertEqual(M.CaseManifest.from_dict(self.case.to_dict()), self.case)


class TestMultiSourceValidationRules(unittest.TestCase):
    """The rules that stop a multi-source case meaning something it does not."""

    def _case(self, **kw):
        return replace(M.get_case(CASE_ID), **kw)

    def test_shares_that_do_not_sum_to_one_are_refused(self):
        bad = self._case(sources=(
            FundedSource("direct_001", SOURCE_DIRECT, 0.5, 0),
            FundedSource("acquired_001", SOURCE_ACQUIRED, 0.2, 1,
                         acquisition_date="2024-12-31")))
        self.assertTrue(any("sum to exactly 1.0" in p
                            for p in M.validate_manifest(bad)))

    def test_duplicate_portfolio_ids_are_refused(self):
        bad = self._case(sources=(
            FundedSource("direct_001", SOURCE_DIRECT, 0.5, 0),
            FundedSource("direct_001", SOURCE_ACQUIRED, 0.5, 1,
                         acquisition_date="2024-12-31")))
        self.assertTrue(any("distinct" in p for p in M.validate_manifest(bad)))

    def test_shared_seed_offsets_are_refused(self):
        bad = self._case(sources=(
            FundedSource("direct_001", SOURCE_DIRECT, 0.5, 0),
            FundedSource("acquired_001", SOURCE_ACQUIRED, 0.5, 0,
                         acquisition_date="2024-12-31")))
        self.assertTrue(any("same book under two labels" in p
                            for p in M.validate_manifest(bad)))

    def test_an_acquired_source_without_an_acquisition_date_is_refused(self):
        with self.assertRaises(ValueError):
            FundedSource("acquired_001", SOURCE_ACQUIRED, 0.5, 1)

    def test_one_provenance_type_only_is_refused(self):
        bad = self._case(sources=(
            FundedSource("direct_001", SOURCE_DIRECT, 0.5, 0),
            FundedSource("direct_002", SOURCE_DIRECT, 0.5, 1)))
        self.assertTrue(any("must carry both" in p
                            for p in M.validate_manifest(bad)))


class TestCombinedTruthIsIndependent(unittest.TestCase):
    """The SPV expectation is the SUM of the parts, computed here."""

    @classmethod
    def setUpClass(cls):
        cls.case = replace(M.get_case(CASE_ID), months=2)
        cls.by_source = {}
        for source in cls.case.funded_sources():
            source_case = cls.case.source_case(source)
            count = max(1, round(cls.case.opening_loan_count * source.share))
            cls.by_source[source.portfolio_id] = build_expected_truth(
                source_case, generate_history(source_case, loan_count=count))
        cls.truth = combine_truths(cls.case, cls.by_source)

    def test_the_spv_balance_is_the_sum_of_its_sources(self):
        for period in self.truth["periods"]:
            parts = sum(v["closing_balance"]
                        for v in period["by_source"].values())
            self.assertAlmostEqual(period["closing_balance"], round(parts, 2),
                                   places=2)

    def test_the_spv_count_is_the_sum_of_its_sources(self):
        for period in self.truth["periods"]:
            parts = sum(v["loan_count"] for v in period["by_source"].values())
            self.assertEqual(period["loan_count"], parts)

    def test_the_sources_are_genuinely_different_books(self):
        balances = {pid: t["periods"][-1]["closing_balance"]
                    for pid, t in self.by_source.items()}
        self.assertEqual(len(set(balances.values())), len(balances), balances)

    def test_weighted_averages_are_recombined_not_averaged(self):
        """A balance-weighted mean recombines exactly; a plain mean does not."""
        period = self.truth["periods"][-1]
        parts = [t["periods"][-1] for t in self.by_source.values()]
        num = sum(p["weighted_average_current_ltv"] * p["closing_balance"]
                  for p in parts)
        den = sum(p["closing_balance"] for p in parts)
        self.assertAlmostEqual(period["weighted_average_current_ltv"],
                               round(num / den, 6), places=4)

    def test_non_combinable_figures_are_omitted_not_approximated(self):
        period = self.truth["periods"][-1]
        for field in ("largest_concentrations", "largest_borrower",
                      "asset_measures"):
            self.assertNotIn(field, period)

    def test_misaligned_reporting_dates_are_refused(self):
        shifted = dict(self.by_source)
        first = next(iter(shifted))
        shifted[first] = dict(shifted[first],
                              reporting_dates=["1999-01-31", "1999-02-28"])
        with self.assertRaises(ValueError):
            combine_truths(self.case, shifted)


class TestObservedModuleReporting(unittest.TestCase):
    """A run summary must report what RAN, never a declared list.

    Regression: ``simulation/pipeline.py`` used to ship a ``PRODUCTION_MODULES``
    tuple that named ``engine.platform_assembler`` on every run, while nothing
    ever invoked it. Reporting an unexecuted module overstates coverage in
    exactly the evidence a reader would use to check coverage.
    """

    def test_there_is_no_declared_production_module_list(self):
        self.assertFalse(hasattr(P, "PRODUCTION_MODULES"))

    def test_observations_start_empty_and_record_evidence(self):
        P.reset_observations()
        self.assertEqual(P.observed_modules(), {})
        P.observe("engine.provenance", "unit test")
        self.assertEqual(P.observed_modules(), {"engine.provenance": "unit test"})
        P.reset_observations()

    def test_an_unknown_module_cannot_be_observed(self):
        with self.assertRaises(ValueError):
            P.observe("not.a.real.module", "unit test")

    def test_every_observed_module_names_a_real_file(self):
        for module in P.KNOWN_PRODUCTION_MODULES:
            parts = module.split(".")
            candidates = [_REPO.joinpath(*parts).with_suffix(".py"),
                          _REPO.joinpath(*parts[:-1]).with_suffix(".py"),
                          _REPO.joinpath(*parts[:-2]).with_suffix(".py")]
            self.assertTrue(any(c.exists() for c in candidates), module)

    def test_a_single_source_run_does_not_claim_the_assembler(self):
        """The exact overstatement that prompted this change."""
        P.reset_observations()
        P.observe("engine.orchestrator.trakt_run", "unit test")
        self.assertNotIn("engine.platform_assembler", P.observed_modules())
        P.reset_observations()


class TestMultiSourceEndToEnd(unittest.TestCase):
    """One small end-to-end run: two deliveries, one assembled SPV."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sim_multi_"))
        case = replace(M.get_case(CASE_ID), months=2,
                       dialects=("clean_csv",))
        cls.result = R.run_case(case, run_root=cls.tmp, scale="small", jobs=3)
        cls.summary = json.loads(
            (Path(cls.result.run_dir) / "run_summary.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_case_passed(self):
        failures = [s.to_dict() for s in self.result.failures]
        self.assertTrue(self.result.ok, json.dumps(failures, indent=1)[:4000])

    def test_the_assembly_stage_ran_and_asserted(self):
        stage = next(s for s in self.result.stages if s.stage == STAGE_ASSEMBLE)
        self.assertFalse(stage.skipped)
        self.assertTrue(stage.assertions)

    def test_the_production_assembler_is_reported_as_executed(self):
        self.assertIn("engine.platform_assembler",
                      self.summary["production_modules"])
        self.assertIn("engine.assembler_agent",
                      self.summary["production_modules"])

    def test_every_reported_module_carries_its_evidence(self):
        evidence = self.summary["production_module_evidence"]
        self.assertEqual(sorted(evidence), sorted(self.summary["production_modules"]))
        self.assertTrue(all(str(v).strip() for v in evidence.values()), evidence)

    def test_the_assembled_canonical_holds_both_deliveries(self):
        import pandas as pd

        case = M.get_case(CASE_ID)
        date = replace(case, months=2).reporting_dates()[-1]
        assembled = (Path(self.result.run_dir) / "platform_assembly"
                     / "clean_csv" / date / P.PLATFORM_CANONICAL_NAME)
        self.assertTrue(assembled.exists(), assembled)
        frame = pd.read_csv(assembled, low_memory=False)
        self.assertEqual(
            sorted(set(frame["source_portfolio_id"].astype(str))),
            sorted(s.portfolio_id for s in case.sources))
        self.assertEqual(
            sorted(set(frame["source_portfolio_type"].astype(str))),
            [SOURCE_ACQUIRED, SOURCE_DIRECT])
        # No duplication: the composite platform key is unique.
        composite = (frame["source_portfolio_id"].astype(str) + "|"
                     + frame["loan_identifier"].astype(str))
        self.assertFalse(composite.duplicated().any())

    def test_the_assembler_run_id_is_deterministic(self):
        manifests = json.loads(
            (Path(self.result.run_dir) / "platform_assembly"
             / "assembler_manifests.json").read_text(encoding="utf-8"))
        for date, manifest in manifests.items():
            self.assertEqual(manifest["assembler_run_id"],
                             f"asm_{CASE_ID}_{date}")


class TestSingleSourceSkipsAssemblyHonestly(unittest.TestCase):
    def test_a_single_source_case_records_the_skip_with_a_reason(self):
        tmp = Path(tempfile.mkdtemp(prefix="sim_single_"))
        try:
            case = replace(M.get_case("bridge_clean_short_duration_v1"),
                           months=2, dialects=("clean_csv",))
            result = R.run_case(case, run_root=tmp, scale="small", jobs=2,
                                stages=["platform_assembly"])
            stage = next(s for s in result.stages if s.stage == STAGE_ASSEMBLE)
            self.assertTrue(stage.skipped)
            self.assertIn("nothing to consolidate", stage.skip_reason)
            self.assertNotIn("engine.platform_assembler", result.production_modules)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
