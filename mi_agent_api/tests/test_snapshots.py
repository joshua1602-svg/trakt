#!/usr/bin/env python3
"""mi_agent_api/tests/test_snapshots.py

The MI landing page is data-driven and deterministic. These tests cover the
snapshot layer that backs it WITHOUT running the heavy onboarding pipeline:

  * portfolio / run discovery from local onboarding output folders;
  * dropdown data only reflects runs that actually exist on disk;
  * deterministic landing-page KPI calculation (balance / count / weighted LTV);
  * month-on-month change (loan-count / balance / new / exited loans);
  * graceful "no prior reporting date available" handling;
  * technical warnings hidden from user-facing output but retained as diagnostics.

A synthetic output tree mirrors the promoted layout
``<root>/<client_id>/<run_id>/output/central/18_central_lender_tape.csv``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent_api import snapshots as S
from mi_agent_api.adapters import split_warnings
from mi_agent_api.data_source import semantics_path

_OCT_N, _OCT_EACH = 33, 127515.15      # -> ~£4.208MM
_NOV_N, _NOV_EACH = 73, 121958.90      # -> ~£8.903MM
_CENTRAL = "18_central_lender_tape.csv"


def _tape_df(n: int, each: float, reporting_date: str) -> pd.DataFrame:
    ids = [760000 + i for i in range(n)]
    return pd.DataFrame({
        "loan_identifier": ids,
        "current_outstanding_balance": [each] * n,
        "current_valuation_amount": [each * 2.0] * n,        # LTV ~ 0.5
        "current_interest_rate": [3.10 + (i % 5) * 0.05 for i in range(n)],
        "current_principal_balance": [each] * n,
        "origination_date": ["2020-06-15"] * n,
        "reporting_date": [reporting_date] * n,
        "data_cut_off_date": [reporting_date] * n,
        "exposure_currency_denomination": ["GBP"] * n,
    })


def _write_run(root: Path, client_id: str, run_id: str, df: pd.DataFrame) -> Path:
    d = root / client_id / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    p = d / _CENTRAL
    df.to_csv(p, index=False)
    return p


class TestDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="snap_disc_"))
        _write_run(cls.root, "client_001", "mi_2025_10", _tape_df(_OCT_N, _OCT_EACH, "2025-10-31"))
        _write_run(cls.root, "client_001", "mi_2025_11", _tape_df(_NOV_N, _NOV_EACH, "2025-11-30"))
        cls.index = S.discover_snapshots(cls.root)

    def test_one_portfolio_two_runs(self):
        pfs = self.index["portfolios"]
        self.assertEqual(len(pfs), 1)
        self.assertEqual(pfs[0]["client_id"], "client_001")
        run_ids = [r["run_id"] for r in pfs[0]["runs"]]
        self.assertEqual(run_ids, ["mi_2025_10", "mi_2025_11"])  # ordered by date

    def test_runs_carry_count_balance_and_date(self):
        runs = {r["run_id"]: r for r in self.index["portfolios"][0]["runs"]}
        self.assertEqual(runs["mi_2025_10"]["loan_count"], 33)
        self.assertEqual(runs["mi_2025_11"]["loan_count"], 73)
        self.assertAlmostEqual(runs["mi_2025_11"]["current_outstanding_balance"],
                               8_902_999.70, delta=2_000)
        self.assertEqual(runs["mi_2025_10"]["reporting_date"], "2025-10-31")
        self.assertEqual(runs["mi_2025_11"]["reporting_date"], "2025-11-30")

    def test_only_available_runs_discovered(self):
        run_ids = {r["run_id"] for r in self.index["portfolios"][0]["runs"]}
        self.assertNotIn("mi_2025_12", run_ids)   # never written -> never offered
        self.assertNotIn("mi_2025_09", run_ids)

    def test_empty_root_is_safe(self):
        empty = Path(tempfile.mkdtemp(prefix="snap_empty_"))
        self.assertEqual(S.discover_snapshots(empty), {"portfolios": []})

    def test_resolve_tape_path(self):
        tape = S.resolve_tape_path(self.root, "client_001", "mi_2025_11")
        self.assertIsNotNone(tape)
        self.assertTrue(str(tape).endswith(_CENTRAL))

    def test_prior_run_resolution(self):
        prior = S.find_prior_run(self.index, "client_001", "mi_2025_11")
        self.assertEqual(prior["run_id"], "mi_2025_10")
        self.assertIsNone(S.find_prior_run(self.index, "client_001", "mi_2025_10"))


class TestSnapshotKPIs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.sem = load_mi_semantics(semantics_path())
        oct_df, oct_rep = cls._prep(_tape_df(_OCT_N, _OCT_EACH, "2025-10-31"))
        cls.oct = S.compute_funded_snapshot(
            oct_df, cls.sem, client_id="client_001", run_id="mi_2025_10",
            reporting_date="2025-10-31", prep_report=oct_rep,
        )
        nov_df, nov_rep = cls._prep(_tape_df(_NOV_N, _NOV_EACH, "2025-11-30"))
        cls.nov = S.compute_funded_snapshot(
            nov_df, cls.sem, client_id="client_001", run_id="mi_2025_11",
            reporting_date="2025-11-30", prep_report=nov_rep,
            prior_df=oct_df, prior_run_id="mi_2025_10",
            prior_reporting_date="2025-10-31",
        )

    @classmethod
    def _prep(cls, df):
        from mi_agent_api.funded_prep import prepare_funded_mi_dataset
        return prepare_funded_mi_dataset(df)

    def _kpis(self, snap):
        return {k["id"]: k for k in snap["kpis"]}

    def test_headline_numbers(self):
        self.assertEqual(self.nov["loan_count"], 73)
        self.assertAlmostEqual(self.nov["current_outstanding_balance"], 8_902_999.70, delta=2_000)
        k = self._kpis(self.nov)
        self.assertEqual(k["loans"]["value"], "73")
        self.assertIn("MM", k["balance"]["value"])

    def test_weighted_average_current_ltv_is_percent(self):
        k = self._kpis(self.nov)
        self.assertTrue(k["wa_current_ltv"]["available"])
        # valuation = 2x balance -> LTV ~ 50%.
        self.assertAlmostEqual(k["wa_current_ltv"]["raw"], 50.0, delta=1.0)
        self.assertTrue(k["wa_current_ltv"]["value"].endswith("%"))

    def test_weighted_average_rate_present(self):
        k = self._kpis(self.nov)
        self.assertIn("wa_rate", k)
        self.assertTrue(k["wa_rate"]["value"].endswith("%"))

    def test_average_loan_balance(self):
        k = self._kpis(self.nov)
        self.assertAlmostEqual(k["avg_balance"]["raw"], _NOV_EACH, delta=1.0)

    def test_monthly_change(self):
        mc = self.nov["monthly_change"]
        self.assertEqual(mc["loan_count_change"], 40)               # +40 loans
        self.assertAlmostEqual(mc["balance_change"], 4_695_000, delta=20_000)  # +£4.7MM
        self.assertGreater(mc["balance_change_pct"], 100.0)
        self.assertEqual(mc["new_loans"], 40)
        self.assertEqual(mc["exited_loans"], 0)
        k = self._kpis(self.nov)
        self.assertEqual(k["mom_loans"]["value"], "+40")
        self.assertTrue(k["mom_balance"]["value"].startswith("+"))

    def test_no_prior_run(self):
        self.assertIsNone(self.oct["monthly_change"])
        self.assertIsNone(self.oct["prior"])
        self.assertTrue(any("No prior reporting date available" in d
                            for d in self.oct["diagnostics"]))
        # No month-on-month KPI tiles when there is no prior run.
        ids = {k["id"] for k in self.oct["kpis"]}
        self.assertNotIn("mom_loans", ids)


class TestWarningClassification(unittest.TestCase):
    def test_technical_warnings_hidden_business_retained(self):
        raw = [
            "percent-scale heuristically detected as 'whole_number_percent' "
            "(median 1.844); the executor does NOT rescale percentages",
            "No chart rendered (chart_type='none'); showing the result table only.",
            "filter current_loan_to_value > 0.8 kept 12/73 rows",
            "current LTV is unavailable for this run",          # business
            "heatmap could not be rendered; showing the result table.",  # business
        ]
        business, diagnostics = split_warnings(raw)
        self.assertIn("current LTV is unavailable for this run", business)
        self.assertIn("heatmap could not be rendered; showing the result table.", business)
        joined = " ".join(diagnostics)
        self.assertIn("percent-scale heuristically detected", joined)
        self.assertIn("No chart rendered", joined)
        # The technical noise must NOT leak into the user-facing list.
        self.assertFalse(any("percent-scale" in b for b in business))


class TestSnapshotEndpoints(unittest.TestCase):
    """The /mi/snapshots + /mi/snapshot routes drive the data-driven dropdowns
    and landing-page snapshot directly from local onboarding output folders."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="snap_api_"))
        _write_run(cls.root, "client_001", "mi_2025_10", _tape_df(_OCT_N, _OCT_EACH, "2025-10-31"))
        _write_run(cls.root, "client_001", "mi_2025_11", _tape_df(_NOV_N, _NOV_EACH, "2025-11-30"))

    def setUp(self):
        import os
        from mi_agent_api import data_source
        for k in ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID",
                  "MI_AGENT_DATA_CSV"):
            os.environ.pop(k, None)
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        data_source.reset_cache()

    def tearDown(self):
        import os
        os.environ.pop("MI_AGENT_ONBOARDING_OUTPUT_ROOT", None)

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def test_snapshots_lists_only_available_runs(self):
        body = self._client().get("/mi/snapshots").json()
        self.assertEqual(len(body["portfolios"]), 1)
        run_ids = [r["run_id"] for r in body["portfolios"][0]["runs"]]
        self.assertEqual(run_ids, ["mi_2025_10", "mi_2025_11"])

    def test_snapshot_november_with_change(self):
        body = self._client().get("/mi/snapshot?portfolioId=client_001/mi_2025_11").json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["loan_count"], 73)
        self.assertEqual(body["monthly_change"]["loan_count_change"], 40)
        self.assertAlmostEqual(body["monthly_change"]["balance_change"], 4_695_000, delta=20_000)
        self.assertEqual(body["prior"]["run_id"], "mi_2025_10")

    def test_snapshot_october_no_prior(self):
        body = self._client().get("/mi/snapshot?portfolioId=client_001/mi_2025_10").json()
        self.assertTrue(body["ok"])
        self.assertIsNone(body["monthly_change"])
        self.assertTrue(any("No prior reporting date available" in d
                            for d in body["diagnostics"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
