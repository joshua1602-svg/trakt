#!/usr/bin/env python3
"""`POST /mi/query` answers or refuses. It does not fall over.

`execute_governed_mi_query` promises, in its own docstring, "a controlled
GovernedResult with a typed error ... so a channel can never turn a governed
failure into a plausible narrative". That promise rested on three assumptions
that were not true:

  * the ANALYTICAL section had no `try` at all. The post-routing guards run
    outside the routing try/except whose comment reads "routing must never break
    the chat", so a fault in any of the six produced a raw 500 — a dead endpoint
    where the contract says `ok: false`;
  * GOVERNANCE caught only `TraktError`, while the block immediately below it
    caught every exception. A tenant registry raising `KeyError` therefore
    escaped with no envelope, no audit event and no request id;
  * DATASET RESOLUTION ran before both, outside every net.

Each is pinned below by injecting the fault and asserting on the status the
React client actually receives. These fail on the pre-fix code.
"""
from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from mi_agent_api import mi_service
from mi_agent_api.app import app
from trakt_core.errors import ErrorCode

QUESTION = "Give me a concise overview of the funded portfolio."


def _ask(client: TestClient):
    return client.post("/mi/query", json={"question": QUESTION})


class TestAnalyticalFaultsRefuse(unittest.TestCase):
    """An analytical fault is a refusal the reader can act on, not a 500."""

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_an_analysis_fault_is_a_controlled_refusal(self):
        with mock.patch.object(mi_service, "_run_analysis",
                               side_effect=ValueError("boom")):
            r = _ask(self.client)
        self.assertEqual(r.status_code, 200,
                         "an analytical fault reached the client as HTTP %s"
                         % r.status_code)
        body = r.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["governance"]["error"]["code"],
                         ErrorCode.CALCULATION_FAILED)
        # The cause is logged, never published.
        self.assertNotIn("boom", body.get("error") or "")

    def test_every_post_routing_guard_is_inside_the_net(self):
        """The six that were measured producing 500s, one at a time."""
        for name in ("_stamp_routed_scope", "_guard_routed_answer",
                     "_guard_temporal_honouring", "_guard_unresolved_scope",
                     "_guard_unknown_category", "_governed_context"):
            with self.subTest(guard=name):
                with mock.patch.object(mi_service, name,
                                       side_effect=ValueError("boom")):
                    r = _ask(self.client)
                self.assertEqual(
                    r.status_code, 200,
                    "%s raising produced HTTP %s" % (name, r.status_code))
                self.assertFalse(r.json()["ok"])


class TestGovernanceFaultsAreTyped(unittest.TestCase):
    """Entitlement that cannot be established refuses — but as a governed event."""

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_a_non_trakt_error_still_produces_a_governed_envelope(self):
        with mock.patch.object(mi_service, "authorise_portfolio_access",
                               side_effect=KeyError("tenant registry entry")):
            r = _ask(self.client)
        body = r.json()
        # It REFUSES — answering around a governance fault would be the unsafe
        # outcome, and the status says so. What must not happen is escaping
        # untyped and unaudited.
        self.assertFalse(body["ok"])
        self.assertEqual(body["governance"]["error"]["code"],
                         ErrorCode.INTERNAL_ERROR)
        self.assertTrue(body["governance"]["requestId"],
                        "a governance fault must be traceable to a request id")
        self.assertNotIn("tenant registry entry", body.get("error") or "")

    def test_a_typed_refusal_keeps_its_own_status(self):
        """The broad catch must not swallow the codes that were already right."""
        from trakt_core.errors import TraktError

        with mock.patch.object(
                mi_service, "authorise_portfolio_access",
                side_effect=TraktError(ErrorCode.PORTFOLIO_NOT_AUTHORISED,
                                       "not yours")):
            r = _ask(self.client)
        self.assertEqual(r.status_code, 403)


class TestDatasetResolutionCannotFailTheRequest(unittest.TestCase):
    """Classifying the sentence runs before every net; it must not need one."""

    def test_a_resolution_fault_falls_back_to_the_default_view(self):
        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch.object(mi_service.workspace_mod, "resolve_dataset",
                               side_effect=ValueError("boom")):
            r = _ask(client)
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
