#!/usr/bin/env python3
"""Sprint 2 — exposing the governed stage-transition capability.

This sprint added no analytical code. What it must prove is therefore not that
the numbers are right (the engine sprint did that) but that BOTH consumers read
the SAME governed object through the architecture that already existed:

  * NO new HTTP route — the movement-detail route was already parameterised by
    detail type, and that is the extension point used;
  * the route returns the engine payload VERBATIM, not a reshaped copy;
  * the deck's shared ``DashboardData`` carries the same payload, resolved
    through the same governed function;
  * the engine's TYPED unavailability reaches both consumers, so neither
    decides availability for itself;
  * the committed React fixture is still the engine's real output, so the
    component test cannot drift into asserting a fiction;
  * no MI Query file was touched.
"""

from __future__ import annotations

import json
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

ROUTE = "/mi/insight/movement-detail"
TRANSITION_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "pipeline_transition_2w"
REACT_FIXTURE = (_REPO_ROOT / "frontend" / "mi-agent-ui" / "src" / "test"
                 / "fixtures" / "stageTransitionDetail.json")


def _funded_df() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(5)],
        "current_outstanding_balance": [100_000.0 + i for i in range(5)],
        "geographic_region_obligor": ["London"] * 5,
        "broker_channel": ["Air"] * 5,
    })


class _ApiBase(unittest.TestCase):
    """The same harness the existing movement-detail endpoint tests use."""

    PIPELINE = TRANSITION_FIXTURE

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        if not cls.PIPELINE.exists():
            raise unittest.SkipTest("transition fixture pack not present")
        cls.root = Path(tempfile.mkdtemp(prefix="stx_api_"))
        central = cls.root / "client_001" / "mi_2025_11" / "output" / "central"
        central.mkdir(parents=True)
        _funded_df().to_csv(central / "18_central_lender_tape.csv", index=False)

    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(self.PIPELINE)
        os.environ[md.FLAG_ENV] = "true"

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT",
                  md.FLAG_ENV):
            os.environ.pop(k, None)

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def _get(self, **params):
        params.setdefault("detailType", md.DETAIL_STAGE_TRANSITION)
        params.setdefault("portfolioId", "client_001/mi_2025_11")
        return self._client().get(ROUTE, params=params)

    def _engine(self) -> Dict[str, Any]:
        # The route resolves "<client>/<run>" to the client id before calling
        # the engine, so the direct call must use the same resolved id.
        return md.resolve_stage_transition_detail(str(self.PIPELINE), "client_001")


# --------------------------------------------------------------------------- #
# 1. The acceptance criterion: no new route.
# --------------------------------------------------------------------------- #
class TestNoNewRoute(_ApiBase):

    def _paths(self):
        from mi_agent_api.app import app
        return {str(getattr(r, "path", "")) for r in app.routes}

    def test_no_new_http_route_was_added_for_the_capability(self):
        """NEW HTTP ROUTES ADDED: 0. The capability is served by the route that
        already owned two-snapshot movement, under a third detail type."""
        offenders = sorted(p for p in self._paths()
                           if "transition" in p.lower() or "stage-movement" in p.lower())
        self.assertEqual(offenders, [],
                         f"a bespoke stage-transition route was added: {offenders}")

    def test_the_existing_movement_route_is_the_owner(self):
        self.assertIn(ROUTE, self._paths())
        self.assertEqual(self._get().status_code, 200)

    def test_there_is_exactly_one_movement_endpoint(self):
        self.assertEqual([p for p in self._paths() if "movement" in p], [ROUTE])

    def test_the_capability_rides_the_existing_feature_flag(self):
        """Not a second flag: the same switch that gates the movement layer."""
        os.environ.pop(md.FLAG_ENV, None)
        self.assertEqual(self._get().status_code, 404)

    def test_an_unknown_detail_type_is_still_rejected(self):
        self.assertEqual(self._get(detailType="PIPELINE_SANKEY").status_code, 400)

    def test_the_two_existing_detail_types_are_untouched(self):
        for dt in (md.DETAIL_PIPELINE, md.DETAIL_COMPLETIONS):
            body = self._get(detailType=dt).json()
            self.assertEqual(body["detail_type"], dt)
            # Still the NET contract, with its own keys — unchanged in shape.
            self.assertIn("contributors", body)
            self.assertIn("components", body)
            self.assertNotIn("transitions", body)


# --------------------------------------------------------------------------- #
# 2. The route returns the ENGINE payload, not a reshaped copy.
# --------------------------------------------------------------------------- #
class TestRouteServesTheEngineResult(_ApiBase):

    def test_the_payload_equals_resolve_stage_transition_detail(self):
        """Field for field. A route that reshaped the result could disagree with
        the deck rendering the same window."""
        body = self._get().json()
        engine = self._engine()
        body.pop("portfolioScope", None)
        self.assertEqual(body, engine)

    def test_every_governed_block_survives_the_hop(self):
        body = self._get().json()
        for key in ("detail_type", "available", "identifier", "measure",
                    "counts", "transitions", "new_arrivals", "stayers",
                    "departures", "event_totals", "reconciliation",
                    "methodology", "source_dates", "sources"):
            self.assertIn(key, body, f"{key} lost between engine and route")

    def test_the_reconciliation_and_residuals_are_not_flattened_away(self):
        recon = self._get().json()["reconciliation"]
        self.assertEqual(recon["count_reconciliation_residual"], 0)
        self.assertEqual(recon["amount_reconciliation_residual"], 0.0)
        self.assertEqual(len(recon["by_stage"]), 5)
        self.assertEqual(recon["global"]["residual"], 0)

    def test_the_transition_matrix_is_the_engines(self):
        self.assertEqual(
            [(t["source_stage"], t["destination_stage"], t["case_count"])
             for t in self._get().json()["transitions"]],
            [("KFI", "APPLICATION", 2), ("APPLICATION", "OFFER", 2),
             ("OFFER", "COMPLETED", 1)])

    def test_new_arrivals_keep_no_source_stage(self):
        for row in self._get().json()["new_arrivals"]:
            self.assertNotIn("source_stage", row)

    def test_unclassified_departures_reach_the_consumer_unresolved(self):
        outcomes = {d["source_stage"]: d["governed_outcome"]
                    for d in self._get().json()["departures"]}
        self.assertEqual(outcomes["OFFER"], md.UNCLASSIFIED_DEPARTURE)
        self.assertEqual(outcomes["APPLICATION"], md.UNCLASSIFIED_DEPARTURE)
        self.assertEqual(outcomes["COMPLETED"], "COMPLETED")
        self.assertEqual(outcomes["WITHDRAWN"], "WITHDRAWN")


# --------------------------------------------------------------------------- #
# 3. Typed unavailability propagates.
# --------------------------------------------------------------------------- #
class TestUnavailabilityPropagates(_ApiBase):

    def test_the_earliest_snapshot_refuses_in_the_transition_contract(self):
        body = self._get(asOf="2026-06-05").json()
        self.assertFalse(body["available"])
        self.assertEqual(body["reason_code"], md.REASON_NO_COMPARISON)
        # Refused in ITS OWN shape, not the movement envelope a transition
        # consumer cannot read.
        self.assertEqual(body["detail_type"], md.DETAIL_STAGE_TRANSITION)
        self.assertEqual(body["transitions"], [])
        self.assertIsNone(body["reconciliation"])
        self.assertNotIn("contributors", body)

    def test_a_movement_refusal_still_uses_the_movement_envelope(self):
        body = self._get(detailType=md.DETAIL_PIPELINE, asOf="2026-06-05").json()
        self.assertFalse(body["available"])
        self.assertIn("contributors", body)
        self.assertNotIn("reason_code", body)

    def test_the_route_never_500s_on_a_bad_point(self):
        res = self._get(asOf="1999-01-01")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["available"])


# --------------------------------------------------------------------------- #
# 4. The React fixture is the engine's real output.
# --------------------------------------------------------------------------- #
class TestReactFixtureIsNotAFiction(unittest.TestCase):
    """The component test asserts against a committed JSON payload. If that file
    could drift from the engine, the React suite would go on passing against a
    contract the backend no longer produces — the exact failure this sprint's
    parity requirement exists to prevent."""

    def test_the_committed_react_fixture_still_equals_the_engine(self):
        if not REACT_FIXTURE.exists():
            self.skipTest("React fixture not present")
        if not TRANSITION_FIXTURE.exists():
            self.skipTest("transition fixture pack not present")
        committed = json.loads(REACT_FIXTURE.read_text(encoding="utf-8"))
        engine = md.resolve_stage_transition_detail(
            str(TRANSITION_FIXTURE), "client_001")
        engine.pop("run_id", None)      # resolution provenance, not analysis
        self.assertEqual(committed, engine,
                         "regenerate with frontend/mi-agent-ui/scripts/"
                         "generate_stage_transition_fixture.py")


# --------------------------------------------------------------------------- #
# 5. Hard guard — MI Query is Sprint 3.
# --------------------------------------------------------------------------- #
class TestNoQueryAgentChange(unittest.TestCase):
    """A structural guard, not a promise in a report: if a later edit teaches the
    Query layer about this capability, this test says so."""

    QUERY_FILES = (
        "mi_agent/llm_query_parser.py",
        "mi_agent_api/chat_routing.py",
        "mi_agent_api/recogniser_registry.py",
        "mi_agent/mi_query_spec.py",
    )

    def test_no_query_file_mentions_the_transition_capability(self):
        for rel in self.QUERY_FILES:
            path = _REPO_ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for token in ("PIPELINE_STAGE_TRANSITION", "stage_transition",
                          "resolve_stage_transition_detail"):
                self.assertNotIn(token, text,
                                 f"{rel} references {token}: Query is Sprint 3")


if __name__ == "__main__":
    unittest.main()
