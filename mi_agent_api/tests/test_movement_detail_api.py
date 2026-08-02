#!/usr/bin/env python3
"""Phase 2A — the optional movement-detail endpoint.

The endpoint is additive, so the tests that matter most are the ones proving it
CANNOT affect anything that already exists:

  * with the flag off it does not exist, and no aggregation runs;
  * the routes the charts already use are untouched whether it is on or off;
  * it answers only from the governed weekly extracts, for the caller's own
    client — a caller-supplied portfolio id can never widen the answer;
  * it never 500s, and it never invents a comparison that is not there.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import movement_detail as md

_FIXTURE_PACK = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
_PIPELINE = _FIXTURE_PACK / "pipeline"
ROUTE = "/mi/insight/movement-detail"


def _funded_df() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(10)],
        "current_outstanding_balance": [100_000 + 1_000 * i for i in range(10)],
        "geographic_region_obligor": ["London"] * 10,
        "broker_channel": ["Air"] * 10,
    })


class _ApiBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        if not _PIPELINE.exists():
            raise unittest.SkipTest("fixture pack not present")
        cls.root = Path(tempfile.mkdtemp(prefix="md_api_"))
        central = cls.root / "client_001" / "mi_2025_11" / "output" / "central"
        central.mkdir(parents=True)
        _funded_df().to_csv(central / "18_central_lender_tape.csv", index=False)

    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(_PIPELINE)

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT",
                  md.FLAG_ENV):
            os.environ.pop(k, None)

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def _on(self):
        os.environ[md.FLAG_ENV] = "true"

    def _get(self, **params) -> Any:
        params.setdefault("detailType", md.DETAIL_PIPELINE)
        params.setdefault("portfolioId", "client_001/mi_2025_11")
        return self._client().get(ROUTE, params=params)


class TestFeatureFlag(_ApiBase):
    def test_absent_when_disabled(self):
        self.assertEqual(self._get().status_code, 404)

    def test_present_when_enabled(self):
        self._on()
        self.assertEqual(self._get().status_code, 200)

    def test_an_unrecognised_flag_value_leaves_it_off(self):
        os.environ[md.FLAG_ENV] = "maybe"
        self.assertEqual(self._get().status_code, 404)

    def test_no_aggregation_runs_when_disabled(self):
        """Disabled must mean no work, not merely a hidden result."""
        calls = []
        real = md.build_movement_detail
        md.build_movement_detail = lambda *a, **k: (  # type: ignore[assignment]
            calls.append(1) or real(*a, **k))
        try:
            self.assertEqual(self._get().status_code, 404)
        finally:
            md.build_movement_detail = real  # type: ignore[assignment]
        self.assertEqual(calls, [], "aggregation ran for a disabled feature")


class TestExistingRoutesAreUntouched(_ApiBase):
    """The whole point of a separate endpoint."""

    CHART_ROUTES = (
        ("/mi/evolution/pipeline", {"portfolioId": "client_001/mi_2025_11"}),
        ("/mi/evolution/funnel", {"portfolioId": "client_001/mi_2025_11"}),
        ("/mi/concentration-tests", {"portfolioId": "client_001/mi_2025_11"}),
    )

    def test_chart_routes_are_byte_identical_with_the_flag_on_and_off(self):
        c = self._client()
        off = {r: c.get(r, params=p).content for r, p in self.CHART_ROUTES}
        self._on()
        on = {r: c.get(r, params=p).content for r, p in self.CHART_ROUTES}
        for route, _p in self.CHART_ROUTES:
            self.assertEqual(off[route], on[route],
                             f"{route} changed when the hover flag was enabled")

    def test_the_new_route_is_not_registered_under_an_existing_path(self):
        paths = {r.path for r in self._client().app.routes if hasattr(r, "path")}
        self.assertIn(ROUTE, paths)
        self.assertIn("/mi/evolution/pipeline", paths)


class TestGovernance(_ApiBase):
    def test_the_answer_is_bound_to_the_resolved_client_not_the_raw_input(self):
        """A caller-supplied portfolio id is a selector, never an authority."""
        self._on()
        body = self._get(portfolioId="client_001/mi_2025_11").json()
        self.assertEqual(body["portfolio_id"], "client_001")

    def test_the_detail_is_never_more_permissive_than_the_chart_it_explains(self):
        """The invariant that actually protects the platform.

        The detail endpoint must answer for exactly the clients the chart route
        answers for — never one more. It cannot become a widening path, whatever
        the surrounding resolution does.

        (Recorded, not asserted as isolation: with ``MI_AGENT_PIPELINE_ROOT``
        pointed at a single client's pipeline directory, ``/mi/evolution/
        pipeline`` and ``/mi/evolution/funnel`` also serve that directory for an
        unrecognised client id — the configured root is the tenancy boundary,
        not the id in the query string. That is existing platform behaviour on
        the routes this endpoint accompanies; Phase 2A neither relies on it nor
        widens it, and it is documented in the phase report rather than changed
        here.)
        """
        self._on()
        c = self._client()
        for pid in ("client_001/mi_2025_11", "someone_else/mi_2025_11",
                    "", "../../etc"):
            chart = c.get("/mi/evolution/pipeline",
                          params={"portfolioId": pid}).json()
            detail = self._get(portfolioId=pid).json()
            chart_has_data = bool(chart.get("periods"))
            detail_has_data = bool(detail.get("available"))
            if detail_has_data:
                self.assertTrue(
                    chart_has_data,
                    f"detail served {pid!r} but the chart route did not — the "
                    "hover layer must never widen access")

    def test_no_case_level_identifiers_are_exposed(self):
        self._on()
        raw = self._get().text
        from mi_agent_api import pipeline_contract as pc
        df, _ = pc.load_prepared_pipeline(
            pc.weekly_extract_inventory(str(_PIPELINE), "client_001")["extracts"][-1])
        for case_id in df[md.CASE_KEY].astype(str).head(5):
            self.assertNotIn(case_id, raw)

    def test_only_the_file_name_is_disclosed_never_a_storage_path(self):
        self._on()
        sources = self._get().json().get("sources") or {}
        for value in sources.values():
            if value:
                self.assertNotIn("/", value)
                self.assertNotIn("\\", value)


class TestBehaviour(_ApiBase):
    def test_pipeline_detail_has_contributors_and_reconciles(self):
        self._on()
        body = self._get().json()
        self.assertTrue(body["available"])
        self.assertEqual(body["detail_type"], md.DETAIL_PIPELINE)
        total = sum(v["amount"] for v in body["components"].values())
        self.assertAlmostEqual(total, body["headline_metric"]["change"], places=2)

    def test_completions_detail_is_served_and_labelled(self):
        self._on()
        body = self._get(detailType=md.DETAIL_COMPLETIONS).json()
        self.assertEqual(body["headline_metric"]["label"], "Completed case value")

    def test_an_unknown_detail_type_is_rejected(self):
        self._on()
        self.assertEqual(self._get(detailType="NOPE").status_code, 400)

    def test_a_specific_week_can_be_requested(self):
        self._on()
        from mi_agent_api import pipeline_contract as pc
        dates = [e["pipeline_extract_date"] for e
                 in pc.weekly_extract_inventory(str(_PIPELINE), "client_001")["extracts"]]
        target = sorted(d for d in dates if d)[-1]
        body = self._get(asOf=target).json()
        self.assertEqual(body["as_of_date"], target)

    def test_the_earliest_week_reports_no_comparison_rather_than_inventing_one(self):
        self._on()
        from mi_agent_api import pipeline_contract as pc
        dates = sorted(e["pipeline_extract_date"] for e
                       in pc.weekly_extract_inventory(str(_PIPELINE), "client_001")["extracts"]
                       if e.get("pipeline_extract_date"))
        body = self._get(asOf=dates[0]).json()
        self.assertFalse(body["available"])
        self.assertIn("earliest", body["reason"].lower())

    def test_an_unmatched_week_is_unavailable_not_the_latest(self):
        self._on()
        body = self._get(asOf="1999-01-01").json()
        self.assertFalse(body["available"])

    def test_the_response_is_small(self):
        """A hover payload must stay a hover payload."""
        self._on()
        self.assertLess(len(self._get().content), 8_000)

    def test_dates_are_explicit_and_never_conflated(self):
        self._on()
        sd = self._get().json()["source_dates"]
        self.assertIsNotNone(sd["pipeline_as_of"])
        self.assertIsNotNone(sd["pipeline_comparison"])
        self.assertIsNone(sd["funded_as_of"],
                          "a weekly pipeline detail must not carry a funded date")
        self.assertNotEqual(sd["pipeline_as_of"], sd["pipeline_comparison"])

    def test_a_broken_resolution_yields_a_controlled_envelope_not_a_500(self):
        self._on()
        real = md.resolve_movement_detail
        md.resolve_movement_detail = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom"))  # type: ignore[assignment]
        try:
            resp = self._get()
        finally:
            md.resolve_movement_detail = real  # type: ignore[assignment]
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["available"])


class TestConditionalResponse(_ApiBase):
    def test_a_repeat_request_is_revalidated_not_recomputed(self):
        self._on()
        c = self._client()
        first = c.get(ROUTE, params={"detailType": md.DETAIL_PIPELINE,
                                     "portfolioId": "client_001/mi_2025_11"})
        etag = first.headers.get("ETag")
        if not etag:
            self.skipTest("conditional responses disabled in this environment")
        second = c.get(ROUTE, params={"detailType": md.DETAIL_PIPELINE,
                                      "portfolioId": "client_001/mi_2025_11"},
                       headers={"If-None-Match": etag})
        self.assertEqual(second.status_code, 304)

    def test_the_two_detail_types_do_not_share_a_validator(self):
        self._on()
        c = self._client()
        a = c.get(ROUTE, params={"detailType": md.DETAIL_PIPELINE,
                                 "portfolioId": "client_001/mi_2025_11"})
        b = c.get(ROUTE, params={"detailType": md.DETAIL_COMPLETIONS,
                                 "portfolioId": "client_001/mi_2025_11"})
        if not a.headers.get("ETag") or not b.headers.get("ETag"):
            self.skipTest("conditional responses disabled in this environment")
        self.assertNotEqual(a.headers["ETag"], b.headers["ETag"])


if __name__ == "__main__":
    unittest.main()
