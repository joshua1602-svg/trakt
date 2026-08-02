#!/usr/bin/env python3
"""Phase 3A — the Weekly Portfolio Brief endpoint.

The endpoint is additive, so the tests that matter most prove it cannot affect
anything that already exists, and that when it does answer it answers for the
caller's own book with its dates stated.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import insight_config as cfg
from mi_agent_api import insight_engine as eng

_FIXTURE_PACK = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
_PIPELINE = _FIXTURE_PACK / "pipeline"
ROUTE = "/mi/insights/weekly-brief"


def _funded() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(10)],
        "current_outstanding_balance": [100_000 + 1_000 * i for i in range(10)],
        "geographic_region_obligor": ["London"] * 10,
        "broker_channel": ["Air"] * 10,
        "current_loan_to_value": [0.3] * 10,
    })


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        if not _PIPELINE.exists():
            raise unittest.SkipTest("fixture pack not present")
        cls.root = Path(tempfile.mkdtemp(prefix="wb_api_"))
        central = cls.root / "client_001" / "mi_2025_11" / "output" / "central"
        central.mkdir(parents=True)
        _funded().to_csv(central / "18_central_lender_tape.csv", index=False)

    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(_PIPELINE)
        cfg.reset_cache()

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT",
                  eng.FLAG_ENV, cfg.PATH_ENV):
            os.environ.pop(k, None)
        cfg.reset_cache()

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def _on(self):
        os.environ[eng.FLAG_ENV] = "true"

    def _get(self, **params) -> Any:
        params.setdefault("portfolioId", "client_001/mi_2025_11")
        return self._client().get(ROUTE, params=params)


class TestFeatureFlag(_Base):
    def test_absent_when_disabled(self):
        self.assertEqual(self._get().status_code, 404)

    def test_present_when_enabled(self):
        self._on()
        self.assertEqual(self._get().status_code, 200)

    def test_an_unrecognised_flag_value_leaves_it_off(self):
        os.environ[eng.FLAG_ENV] = "perhaps"
        self.assertEqual(self._get().status_code, 404)

    def test_no_insight_is_generated_when_disabled(self):
        """Disabled must mean no work, not a hidden result."""
        calls = []
        real = eng.build
        eng.build = lambda *a, **k: (calls.append(1) or real(*a, **k))
        try:
            self.assertEqual(self._get().status_code, 404)
        finally:
            eng.build = real
        self.assertEqual(calls, [], "the engine ran for a disabled feature")


class TestExistingRoutesAreUntouched(_Base):
    ROUTES = (
        ("/mi/evolution/pipeline", {"portfolioId": "client_001/mi_2025_11"}),
        ("/mi/evolution/funnel", {"portfolioId": "client_001/mi_2025_11"}),
        ("/mi/concentration-tests", {"portfolioId": "client_001/mi_2025_11"}),
        ("/mi/snapshot", {"portfolioId": "client_001/mi_2025_11"}),
    )

    def test_chart_routes_are_byte_identical_with_the_flag_on_and_off(self):
        c = self._client()
        off = {r: c.get(r, params=p).content for r, p in self.ROUTES}
        self._on()
        on = {r: c.get(r, params=p).content for r, p in self.ROUTES}
        for route, _p in self.ROUTES:
            self.assertEqual(off[route], on[route],
                             f"{route} changed when the brief flag was enabled")


class TestGovernance(_Base):
    def test_the_brief_is_bound_to_the_resolved_client(self):
        self._on()
        body = self._get().json()
        self.assertEqual(body["portfolio_id"], "client_001")
        self.assertEqual(body["tenant_id"], "client_001")

    def test_the_brief_is_never_more_permissive_than_the_chart_route(self):
        self._on()
        c = self._client()
        for pid in ("client_001/mi_2025_11", "someone_else/mi_2025_11", ""):
            chart = c.get("/mi/evolution/pipeline", params={"portfolioId": pid}).json()
            brief = self._get(portfolioId=pid).json()
            if brief.get("insights"):
                self.assertTrue(
                    bool(chart.get("periods")),
                    f"brief answered for {pid!r} where the chart route did not")

    def test_no_case_level_identifier_is_exposed(self):
        self._on()
        raw = self._get().text
        from mi_agent_api import pipeline_contract as pc
        df, _ = pc.load_prepared_pipeline(
            pc.weekly_extract_inventory(str(_PIPELINE), "client_001")["extracts"][-1])
        for case_id in df["pipeline_case_identifier"].astype(str).head(5):
            self.assertNotIn(case_id, raw)


class TestBehaviour(_Base):
    def test_a_normal_brief_names_every_type_it_did_not_produce(self):
        self._on()
        b = self._get().json()
        self.assertEqual(b["status"], "success")
        accounted = ({i["insight_type"] for i in b["insights"]}
                     | {o["insight_type"] for o in b["omitted"]})
        for t in ("PIPELINE_MOVEMENT", "TICKET_SIZE", "WEIGHTED_LTV",
                  "COMPLETIONS_MOVEMENT", "CONVERSION_CONTEXT",
                  "CONCENTRATION_PROXIMITY", "DATA_QUALITY"):
            self.assertIn(t, accounted)

    def test_never_more_than_eight_insights(self):
        self._on()
        self.assertLessEqual(self._get().json()["insight_count"], 8)

    def test_no_material_insights_is_a_success_not_an_error(self):
        """Silence must be a statement, not a failure."""
        self._on()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            # Impossible thresholds -> nothing can qualify.
            fh.write("insights:\n"
                     "  pipeline_movement:\n    min_change_pct: 100000.0\n"
                     "    min_change_share_of_pipeline_pct: 100000.0\n"
                     "  ticket_size:\n    min_average_change_pct: 100000.0\n"
                     "    min_median_change_pct: 100000.0\n"
                     "  weighted_ltv:\n    min_change_pp: 100000.0\n"
                     "  ticket_mix:\n    min_share_change_pp: 100000.0\n"
                     "  ltv_mix:\n    min_share_change_pp: 100000.0\n"
                     "  completions:\n    min_change_pct: 100000.0\n"
                     "  concentration:\n    amber_utilisation_pct: 100000.0\n"
                     "  conversion:\n    require_sufficient: false\n"
                     "  data_quality:\n"
                     "    max_dimension_reassignment_pct: 100000.0\n"
                     "    max_unknown_share_pct: 100000.0\n"
                     "    max_unusable_id_share_pct: 100000.0\n"
                     "    max_duplicate_id_share_pct: 100000.0\n")
            os.environ[cfg.PATH_ENV] = fh.name
        cfg.reset_cache()
        b = self._get().json()
        self.assertEqual(b["status"], "success")
        self.assertEqual(b["insight_count"], 0)
        self.assertTrue(b["omitted"], "an empty brief must still explain itself")

    def test_the_earliest_week_reports_no_comparison(self):
        self._on()
        from mi_agent_api import pipeline_contract as pc
        dates = sorted(e["pipeline_extract_date"] for e
                       in pc.weekly_extract_inventory(str(_PIPELINE),
                                                      "client_001")["extracts"]
                       if e.get("pipeline_extract_date"))
        b = self._get(asOf=dates[0]).json()
        self.assertIsNone(b["comparison_date"])
        self.assertIn("prior week",
                      " ".join(o["reason"] for o in b["omitted"]).lower())

    def test_an_unmatched_week_is_unavailable_not_the_latest(self):
        self._on()
        b = self._get(asOf="1999-01-01").json()
        self.assertEqual(b["status"], "unavailable")

    def test_the_response_stays_small(self):
        self._on()
        self.assertLess(len(self._get().content), 32_000)

    def test_dates_are_explicit_and_never_conflated(self):
        self._on()
        b = self._get().json()
        self.assertIsNotNone(b["source_dates"]["pipeline_as_of"])
        self.assertNotEqual(b["source_dates"]["pipeline_as_of"],
                            b["source_dates"]["pipeline_comparison"])

    def test_the_configuration_source_is_disclosed(self):
        self._on()
        self.assertTrue(self._get().json()["config_source"])

    def test_a_broken_engine_yields_a_controlled_envelope_not_a_500(self):
        self._on()
        real = eng.build
        eng.build = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            r = self._get()
        finally:
            eng.build = real
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "unavailable")

    def test_the_same_request_twice_gives_the_same_brief(self):
        self._on()
        a, b = self._get().json(), self._get().json()
        self.assertEqual([i["insight_id"] for i in a["insights"]],
                         [i["insight_id"] for i in b["insights"]])


class TestConditionalResponse(_Base):
    def test_a_repeat_request_revalidates_to_304(self):
        self._on()
        c = self._client()
        first = c.get(ROUTE, params={"portfolioId": "client_001/mi_2025_11"})
        etag = first.headers.get("ETag")
        if not etag:
            self.skipTest("conditional responses disabled in this environment")
        second = c.get(ROUTE, params={"portfolioId": "client_001/mi_2025_11"},
                       headers={"If-None-Match": etag})
        self.assertEqual(second.status_code, 304)

    def test_two_weeks_do_not_share_a_validator(self):
        self._on()
        c = self._client()
        a = c.get(ROUTE, params={"portfolioId": "client_001/mi_2025_11"})
        b = c.get(ROUTE, params={"portfolioId": "client_001/mi_2025_11",
                                 "asOf": "2025-11-01"})
        if not a.headers.get("ETag") or not b.headers.get("ETag"):
            self.skipTest("conditional responses disabled in this environment")
        self.assertNotEqual(a.headers["ETag"], b.headers["ETag"])


class TestWarmCost(_Base):
    def test_a_warm_request_reads_no_source_bytes(self):
        """The brief must never re-read a source file to answer a repeat call."""
        self._on()
        c = self._client()
        c.get(ROUTE, params={"portfolioId": "client_001/mi_2025_11"})  # warm

        from apps.blob_trigger_app import storage as storage_mod
        reads = []
        real = storage_mod.Storage.read_bytes
        storage_mod.Storage.read_bytes = (
            lambda self, uri: reads.append(uri) or real(self, uri))
        try:
            r = c.get(ROUTE, params={"portfolioId": "client_001/mi_2025_11"})
        finally:
            storage_mod.Storage.read_bytes = real
        self.assertEqual(r.status_code, 200)
        self.assertEqual(reads, [], f"warm brief read source bytes: {reads}")


if __name__ == "__main__":
    unittest.main()
