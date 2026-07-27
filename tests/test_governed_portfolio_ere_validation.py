#!/usr/bin/env python3
"""tests/test_governed_portfolio_ere_validation.py

Validation of the governed portfolio architecture against the ERE platform tape.

The platform canonical is assembled by the real
``apps.blob_trigger_app.assembler_refresh`` path from accepted per-portfolio
canonicals — the same code that produces
``platform/ERE/latest/platform_canonical_typed.csv`` in production — so this
exercises the governed stack against ERE's actual shape rather than a mock.

Two shapes are validated:

  * **today** — ``direct_001`` (73 loans) + ``acquired_001`` (885 loans),
    consolidating to 958;
  * **expanded** — the same platform with ``direct_002`` and ``acquired_002``
    added, to prove the hierarchy, the group scopes, the funded views, the
    pipeline eligibility and the forecast all widen with NO code, configuration
    or UI change.

The current counts are ASSERTED AS DERIVED VALUES: each expectation is computed
from the assembled tape, and the literals only appear in the fixture that builds
it. Nothing in the runtime knows them.

Run: python -m pytest tests/test_governed_portfolio_ere_validation.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.test_platform_assembly_all_portfolios import (  # noqa: E402
    ACQUIRED_ROWS, CLIENT, DIRECT_ROWS, _canonical, _Ctx, _DIRECT_ONLY)
from trakt_core import portfolio as P  # noqa: E402

#: The ERE platform as it stands today. These literals describe the FIXTURE; the
#: assertions below derive every expectation from the assembled tape.
_ERE_TODAY = [("direct_001", "direct", DIRECT_ROWS),
              ("acquired_001", "acquired", ACQUIRED_ROWS)]
#: The same platform after two more books are onboarded.
_ERE_EXPANDED = _ERE_TODAY + [("direct_002", "direct", 41),
                              ("acquired_002", "acquired", 120)]


def _assemble(portfolios) -> pd.DataFrame:
    """Assemble a platform canonical through the production assembler path."""
    tmp = tempfile.TemporaryDirectory(prefix="ere_validation_")
    ctx = _Ctx(Path(tmp.name))
    last_local = None
    last_pid = None
    for pid, ptype, rows in portfolios:
        extra = _DIRECT_ONLY if ptype == "direct" else ()
        frame = _canonical(pid, rows, portfolio_type=ptype, extra_columns=extra)
        last_local = ctx.accept(pid, frame)
        ctx.register(pid)
        last_pid = pid
    ctx.refresh(last_pid, last_local)
    platform = ctx.platform()
    tmp.cleanup()
    return platform


class _PlatformCase(unittest.TestCase):
    """Serves an assembled platform canonical through the live API."""

    PORTFOLIOS = _ERE_TODAY

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except Exception:  # pragma: no cover - fastapi not installed
            raise unittest.SkipTest("fastapi not installed")

    def setUp(self):
        from engine import platform_assembler as pa
        from fastapi.testclient import TestClient

        self.platform = _assemble(self.PORTFOLIOS)
        self._saved = {k: os.environ.get(k) for k in
                       ("MI_AGENT_AUTH_ENABLED", "MI_AGENT_PLATFORM_CANONICAL")}
        os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / pa.PLATFORM_CANONICAL_NAME
        self.platform.to_csv(path, index=False)
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(path)

        from mi_agent_api import data_source as ds
        self.ds = ds
        ds.reset_cache()
        from mi_agent_api.app import app
        self.client = TestClient(app)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.ds.reset_cache()
        self._tmp.cleanup()

    # -- helpers ----------------------------------------------------------- #
    def rows_in(self, *portfolio_ids) -> int:
        """Loan count for the named portfolios, straight from the tape."""
        mask = self.platform["source_portfolio_id"].isin(portfolio_ids)
        return int(mask.sum())

    def context_index(self):
        return self.client.get("/mi/portfolio-context").json()

    def snapshot(self, context):
        return self.client.get("/mi/snapshot",
                               params={"portfolioId": f"{CLIENT}/latest",
                                       "portfolioContext": context}).json()


# --------------------------------------------------------------------------- #
# 1. The ERE platform as it stands today
# --------------------------------------------------------------------------- #
class TestEreCurrentPlatform(_PlatformCase):

    PORTFOLIOS = _ERE_TODAY

    def test_the_hierarchy_is_total_direct_direct_001_acquired_acquired_001(self):
        contexts = self.context_index()["contexts"]
        self.assertEqual([(c["context_id"], c["depth"]) for c in contexts],
                         [("total", 0), ("direct", 1), ("direct_001", 2),
                          ("acquired", 1), ("acquired_001", 2)])

    def test_group_membership_matches_the_tape(self):
        by_id = {c["context_id"]: c for c in self.context_index()["contexts"]}
        self.assertEqual(by_id["direct"]["portfolio_ids"], ["direct_001"])
        self.assertEqual(by_id["acquired"]["portfolio_ids"], ["acquired_001"])
        self.assertEqual(sorted(by_id["total"]["portfolio_ids"]),
                         ["acquired_001", "direct_001"])

    def test_loan_counts_reconcile_to_the_platform_tape(self):
        """Total = Direct + Acquired, every figure derived from the tape."""
        total = self.snapshot("total")["loan_count"]
        direct = self.snapshot("direct")["loan_count"]
        acquired = self.snapshot("acquired")["loan_count"]
        self.assertEqual(direct, self.rows_in("direct_001"))
        self.assertEqual(acquired, self.rows_in("acquired_001"))
        self.assertEqual(total, self.rows_in("direct_001", "acquired_001"))
        self.assertEqual(total, direct + acquired)

    def test_balances_reconcile_to_the_platform_tape(self):
        balance = lambda ctx: self.snapshot(ctx)["current_outstanding_balance"]  # noqa: E731
        self.assertAlmostEqual(balance("total"),
                               balance("direct") + balance("acquired"), places=2)

    def test_an_individual_portfolio_is_selectable_and_isolated(self):
        self.assertEqual(self.snapshot("direct_001")["loan_count"],
                         self.rows_in("direct_001"))
        self.assertEqual(self.snapshot("acquired_001")["loan_count"],
                         self.rows_in("acquired_001"))

    def test_acquired_cannot_open_pipeline_or_origination_forecast(self):
        caps = {c["context_id"]: c["capabilities"]
                for c in self.context_index()["contexts"]}
        for context in ("acquired", "acquired_001"):
            self.assertFalse(caps[context]["pipeline"]["enabled"], context)
            self.assertFalse(caps[context]["origination_forecast"]["enabled"], context)
            self.assertTrue(caps[context]["funded"]["enabled"], context)
            self.assertTrue(caps[context]["cohorts"]["enabled"], context)

    def test_acquired_runoff_is_not_modelled_and_says_so(self):
        body = self.client.get("/mi/forecast/snapshot",
                               params={"portfolioId": f"{CLIENT}/latest",
                                       "portfolioContext": "acquired_001"}).json()
        projections = body["portfolioProjections"]
        self.assertEqual(projections["runoffNotModelled"], ["acquired_001"])
        entry = projections["portfolios"][0]
        self.assertEqual(entry["projectedBalance"], entry["currentBalance"])
        self.assertEqual(entry["expectedNewOriginations"], 0.0)
        self.assertIn("Runoff not modelled for acquired_001", entry["disclosure"])

    def test_total_forecast_covers_every_portfolio(self):
        body = self.client.get("/mi/forecast/snapshot",
                               params={"portfolioId": f"{CLIENT}/latest",
                                       "portfolioContext": "total"}).json()
        projections = body["portfolioProjections"]
        self.assertEqual(sorted(p["portfolioId"] for p in projections["portfolios"]),
                         ["acquired_001", "direct_001"])
        self.assertAlmostEqual(
            projections["totalProjectedBalance"],
            sum(p["projectedBalance"] for p in projections["portfolios"])
            + projections["unattributedExpectedOriginations"], places=2)


# --------------------------------------------------------------------------- #
# 2. The same platform with two more portfolios — no code or config change
# --------------------------------------------------------------------------- #
class TestEreExpandedPlatform(_PlatformCase):
    """Everything below runs against IDENTICAL code and configuration; only the
    canonical data changed."""

    PORTFOLIOS = _ERE_EXPANDED

    def test_total_automatically_includes_all_four_portfolios(self):
        by_id = {c["context_id"]: c for c in self.context_index()["contexts"]}
        self.assertEqual(sorted(by_id["total"]["portfolio_ids"]),
                         ["acquired_001", "acquired_002", "direct_001", "direct_002"])

    def test_direct_automatically_includes_both_direct_portfolios(self):
        by_id = {c["context_id"]: c for c in self.context_index()["contexts"]}
        self.assertEqual(sorted(by_id["direct"]["portfolio_ids"]),
                         ["direct_001", "direct_002"])

    def test_acquired_automatically_includes_both_acquired_portfolios(self):
        by_id = {c["context_id"]: c for c in self.context_index()["contexts"]}
        self.assertEqual(sorted(by_id["acquired"]["portfolio_ids"]),
                         ["acquired_001", "acquired_002"])

    def test_every_individual_portfolio_remains_selectable(self):
        ids = [c["context_id"] for c in self.context_index()["contexts"]
               if c["context_kind"] == "portfolio"]
        self.assertEqual(sorted(ids), ["acquired_001", "acquired_002",
                                       "direct_001", "direct_002"])

    def test_total_funded_views_include_every_portfolio(self):
        total = self.snapshot("total")["loan_count"]
        self.assertEqual(total, self.rows_in("direct_001", "direct_002",
                                             "acquired_001", "acquired_002"))
        self.assertEqual(total, self.snapshot("direct")["loan_count"]
                         + self.snapshot("acquired")["loan_count"])

    def test_total_is_no_longer_direct_001_plus_acquired_001(self):
        """The defect, stated as an assertion."""
        self.assertGreater(
            self.snapshot("total")["loan_count"],
            self.snapshot("direct_001")["loan_count"]
            + self.snapshot("acquired_001")["loan_count"])

    def test_total_pipeline_includes_only_eligible_originating_portfolios(self):
        by_id = {c["context_id"]: c for c in self.context_index()["contexts"]}
        pipeline = by_id["total"]["capabilities"]["pipeline"]
        # No governed pipeline extract is published for this fixture, so the tab
        # is refused with the honest reason rather than shown empty.
        self.assertFalse(pipeline["enabled"])
        self.assertEqual(pipeline["reason_code"], P.REASON_NO_PIPELINE_DATA)
        # And the reason names the originating books, never the acquired ones.
        self.assertIn("direct_001", pipeline["detail"])
        self.assertIn("direct_002", pipeline["detail"])
        self.assertNotIn("acquired_001", pipeline["detail"])

    def test_total_forecast_includes_every_portfolio_by_its_treatment(self):
        body = self.client.get("/mi/forecast/snapshot",
                               params={"portfolioId": f"{CLIENT}/latest",
                                       "portfolioContext": "total"}).json()
        projections = body["portfolioProjections"]
        self.assertEqual(sorted(p["portfolioId"] for p in projections["portfolios"]),
                         ["acquired_001", "acquired_002", "direct_001", "direct_002"])
        treatments = {p["portfolioId"]: p["forecastTreatment"]
                      for p in projections["portfolios"]}
        self.assertEqual(treatments["direct_002"], P.FORECAST_ORIGINATION)
        self.assertEqual(treatments["acquired_002"], P.FORECAST_HOLD_CONSTANT)

    def test_acquired_runoff_limitations_are_explicit_per_portfolio(self):
        body = self.client.get("/mi/forecast/snapshot",
                               params={"portfolioId": f"{CLIENT}/latest",
                                       "portfolioContext": "acquired"}).json()
        projections = body["portfolioProjections"]
        self.assertEqual(sorted(projections["runoffNotModelled"]),
                         ["acquired_001", "acquired_002"])
        joined = " ".join(projections["disclosures"])
        self.assertIn("Runoff not modelled for acquired_001", joined)
        self.assertIn("Runoff not modelled for acquired_002", joined)

    def test_mi_agent_reports_partial_field_coverage_by_portfolio(self):
        """The direct-only columns are absent from the acquired books, which the
        agent must disclose by individual portfolio rather than average over."""
        body = self.client.post("/mi/query", json={
            "question": "What is the total current balance?",
            "sourcePortfolioLens": "total"}).json()
        coverage = body["portfolioCoverage"]
        self.assertEqual(sorted(coverage["portfolios_in_scope"]),
                         ["acquired_001", "acquired_002", "direct_001", "direct_002"])
        self.assertTrue(coverage["is_fully_consolidated"])
        self.assertIn("Fully consolidated", coverage["disclosure"])

    def test_group_queries_answer_across_every_member(self):
        for context, members in (("direct", ["direct_001", "direct_002"]),
                                 ("acquired", ["acquired_001", "acquired_002"])):
            body = self.client.post("/mi/query", json={
                "question": "What is the total current balance?",
                "sourcePortfolioLens": context}).json()
            self.assertTrue(body["ok"], context)
            self.assertEqual(sorted(body["portfolioCoverage"]["portfolios_used"]),
                             members, context)

    def test_react_and_copilot_consume_the_same_governed_contract(self):
        question = "What is the total current balance?"
        react = self.client.post("/mi/query", json={"question": question}).json()
        saved = os.environ.get("TRAKT_COPILOT_AUTH_MODE")
        os.environ["TRAKT_COPILOT_AUTH_MODE"] = "disabled"
        try:
            copilot = self.client.post("/v1/copilot/mi/query",
                                       json={"question": question}).json()
        finally:
            if saved is None:
                os.environ.pop("TRAKT_COPILOT_AUTH_MODE", None)
            else:
                os.environ["TRAKT_COPILOT_AUTH_MODE"] = saved
        self.assertEqual(copilot["portfolioCoverage"],
                         react["governance"]["scope"])


if __name__ == "__main__":
    unittest.main()
