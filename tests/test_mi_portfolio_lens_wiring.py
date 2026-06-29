#!/usr/bin/env python3
"""tests/test_mi_portfolio_lens_wiring.py

Source-portfolio lens wired into the live MI Agent path (#1), plus the API
discovery endpoint + first-class request field (#2).

Run: python -m unittest tests.test_mi_portfolio_lens_wiring
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

from mi_agent import portfolio_lens as pl
from mi_agent.mi_agent_workflow import run_mi_agent_query

_SEM = str(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml")


def _df():
    return pd.DataFrame({
        "loan_identifier": ["L1", "L2", "L3", "L4"],
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0],
        "source_portfolio_id": ["direct_001", "direct_001", "acquired_001", "acquired_002"],
        "source_portfolio_type": ["direct", "direct", "acquired", "acquired"],
        "source_portfolio_label": ["Direct Book", "Direct Book",
                                   "Acquired Portfolio 1", "Acquired Portfolio 2"],
        "portfolio_cohort": ["direct_001", "direct_001", "acquired_001", "acquired_002"],
    })


class TestLensHelpers(unittest.TestCase):

    def test_mentions_portfolio(self):
        self.assertTrue(pl.mentions_portfolio("show the acquired book"))
        self.assertTrue(pl.mentions_portfolio("direct_001 only"))
        self.assertTrue(pl.mentions_portfolio("total portfolio"))
        self.assertFalse(pl.mentions_portfolio("show balance by region"))

    def test_lens_from_selection(self):
        self.assertEqual(pl.lens_from_selection(None).name, "total")
        self.assertEqual(pl.lens_from_selection("total").name, "total")
        self.assertEqual(pl.lens_from_selection("direct").filters, {"source_portfolio_type": "direct"})
        self.assertEqual(pl.lens_from_selection("acquired").filters, {"source_portfolio_type": "acquired"})
        self.assertEqual(pl.lens_from_selection("acquired_001").filters, {"source_portfolio_id": "acquired_001"})
        self.assertEqual(pl.lens_from_selection({"id": "direct_001"}).name, "cohort")
        self.assertEqual(pl.lens_from_selection("nonsense").name, "total")  # safe fallback

    def test_resolve_with_default_nl_overrides(self):
        # NL names a scope -> overrides the default (dropdown) selection.
        self.assertEqual(pl.resolve_lens_with_default("acquired book", pl.lens_from_selection("direct")).name,
                         "acquired")
        # No scope in NL -> use the default.
        self.assertEqual(pl.resolve_lens_with_default("show balance", pl.lens_from_selection("acquired_001")).name,
                         "cohort")
        # No scope, no default -> total.
        self.assertEqual(pl.resolve_lens_with_default("show balance", None).name, "total")

    def test_is_acquired_only(self):
        self.assertTrue(pl.is_acquired_only(pl.lens_from_selection("acquired")))
        self.assertTrue(pl.is_acquired_only(pl.lens_from_selection("acquired_002")))
        self.assertFalse(pl.is_acquired_only(pl.lens_from_selection("direct")))
        self.assertFalse(pl.is_acquired_only(pl.total_lens()))

    def test_available_lenses(self):
        records = _df()[["source_portfolio_id", "source_portfolio_type",
                         "source_portfolio_label"]].drop_duplicates().to_dict("records")
        lenses = pl.available_lenses(records)
        by_id = {l["id"]: l for l in lenses}
        self.assertEqual(by_id["total"]["funded_only"], False)
        self.assertEqual(by_id["direct"]["funded_only"], False)
        self.assertEqual(by_id["acquired"]["funded_only"], True)
        self.assertEqual(by_id["acquired_001"]["label"], "Acquired Portfolio 1")
        self.assertTrue(by_id["acquired_001"]["funded_only"])
        self.assertFalse(by_id["direct_001"]["funded_only"])

    def test_available_lenses_empty(self):
        self.assertEqual([l["id"] for l in pl.available_lenses([])], ["total"])


class TestWorkflowLensWiring(unittest.TestCase):

    def _lens(self, question, sel=None):
        r = run_mi_agent_query(question, _df(), _SEM, source_portfolio_lens=sel)
        return (r.get("portfolio_lens") or {}), r["spec_obj"].filters, r["ok"]

    def test_nl_direct(self):
        lens, filt, ok = self._lens("show direct book balance")
        self.assertEqual(lens.get("name"), "direct")
        self.assertEqual(filt.get("source_portfolio_type"), "direct")

    def test_nl_acquired_and_backbook(self):
        self.assertEqual(self._lens("show acquired book balance")[0].get("name"), "acquired")
        self.assertEqual(self._lens("show the back book")[0].get("name"), "acquired")

    def test_nl_cohort(self):
        lens, filt, _ = self._lens("balance for acquired_001 only")
        self.assertEqual(lens.get("name"), "cohort")
        self.assertEqual(filt.get("source_portfolio_id"), "acquired_001")

    def test_dropdown_default_applies(self):
        lens, filt, _ = self._lens("show portfolio balance", sel="acquired")
        self.assertEqual(lens.get("name"), "acquired")
        self.assertEqual(filt.get("source_portfolio_type"), "acquired")

    def test_nl_overrides_dropdown(self):
        lens, _, _ = self._lens("show direct book balance", sel="acquired_001")
        self.assertEqual(lens.get("name"), "direct")

    def test_no_provenance_is_noop(self):
        df = pd.DataFrame({"loan_identifier": ["L1"], "current_outstanding_balance": [10.0]})
        r = run_mi_agent_query("total balance", df, _SEM, source_portfolio_lens="acquired")
        self.assertTrue(r["ok"])
        self.assertNotIn("portfolio_lens", r)  # lens block skipped — no degradation


class TestApiLensEndpoints(unittest.TestCase):

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi not installed")
        from engine import platform_assembler as pa
        self._tmp = tempfile.TemporaryDirectory()
        td = Path(self._tmp.name)
        p = td / pa.PLATFORM_CANONICAL_NAME
        _df().to_csv(p, index=False)
        self._saved = os.environ.get("MI_AGENT_PLATFORM_CANONICAL")
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(p)
        from mi_agent_api import data_source as ds
        self.ds = ds
        ds.reset_cache()
        from mi_agent_api.app import app
        self.client = TestClient(app)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("MI_AGENT_PLATFORM_CANONICAL", None)
        else:
            os.environ["MI_AGENT_PLATFORM_CANONICAL"] = self._saved
        self.ds.reset_cache()
        self._tmp.cleanup()

    def test_discovery_endpoint(self):
        r = self.client.get("/mi/source-portfolios").json()
        self.assertTrue(r["available"])
        by_id = {l["id"]: l for l in r["lenses"]}
        self.assertIn("total", by_id)
        self.assertIn("direct", by_id)
        self.assertIn("acquired", by_id)
        self.assertIn("acquired_001", by_id)
        self.assertTrue(by_id["acquired"]["funded_only"])
        self.assertFalse(by_id["direct"]["funded_only"])

    def test_query_with_lens_field(self):
        r = self.client.post("/mi/query",
                             json={"question": "show portfolio balance",
                                   "sourcePortfolioLens": "acquired"}).json()
        lens = (r.get("metadata") or {}).get("portfolioLens") or {}
        self.assertEqual(lens.get("name"), "acquired")

    def test_query_nl_lens(self):
        r = self.client.post("/mi/query",
                             json={"question": "show the acquired book balance"}).json()
        lens = (r.get("metadata") or {}).get("portfolioLens") or {}
        self.assertEqual(lens.get("name"), "acquired")


if __name__ == "__main__":
    unittest.main()
