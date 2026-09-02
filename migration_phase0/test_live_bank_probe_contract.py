#!/usr/bin/env python3
"""Every key the probe sends must be one `/mi/query` actually declares.

WHY THIS FILE EXISTS. `live_bank_probe.py` sent `portfolioContext` to select the
portfolio scope. `QueryRequest` does not declare that field, and pydantic's
default policy for an undeclared key is to DROP IT — no error, no warning, HTTP
200. So `--scope direct` bound nothing and every question ran over the whole
book, including the acquired portfolio at its own reporting date, which is the
precise contamination the flag exists to avoid. The run would have looked
entirely successful and measured the wrong population.

The real field is `sourcePortfolioLens`. A silent drop is not something a probe
can detect from its own output, so it is pinned here instead: the request model
is the authority, and the probe's payload keys are checked against it.
"""
from __future__ import annotations

import unittest

from mi_agent_api.app import QueryRequest

# The keys `live_bank_probe.main` puts in the body. Kept literal on purpose —
# reading them out of the module would just re-import the same mistake.
PROBE_PAYLOAD_KEYS = {"question", "sourcePortfolioLens", "asOfDate"}


class TestProbePayloadIsAccepted(unittest.TestCase):

    def test_every_probe_key_is_declared_by_the_request_model(self):
        declared = set(QueryRequest.model_fields)
        undeclared = PROBE_PAYLOAD_KEYS - declared
        self.assertEqual(
            undeclared, set(),
            "the probe sends %s, which /mi/query does not declare — pydantic "
            "drops it in silence and the run measures something else"
            % sorted(undeclared))

    def test_an_undeclared_scope_key_really_is_dropped_in_silence(self):
        """The failure mode itself, so the reason for this file stays legible."""
        req = QueryRequest(**{"question": "x", "portfolioContext": "direct"})
        self.assertIsNone(req.sourcePortfolioLens,
                          "portfolioContext bound a scope; if the API ever "
                          "accepts it, this test's premise has changed")

    def test_the_field_the_probe_now_sends_binds(self):
        req = QueryRequest(**{"question": "x", "sourcePortfolioLens": "direct"})
        self.assertEqual(req.sourcePortfolioLens, "direct")


class TestTheProbeSendsThatField(unittest.TestCase):
    """Pin the payload in the script itself, not just the set above."""

    def test_the_script_names_the_declared_field(self):
        import pathlib
        src = (pathlib.Path(__file__).parent / "live_bank_probe.py").read_text()
        self.assertIn('"sourcePortfolioLens": args.scope', src)
        self.assertNotIn('"portfolioContext": args.scope', src)


if __name__ == "__main__":
    unittest.main()
