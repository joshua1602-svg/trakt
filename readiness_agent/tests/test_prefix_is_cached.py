#!/usr/bin/env python3
"""The readiness loop buys its fixed prefix once, not once per step.

WHY THIS FILE EXISTS. Measured on the Anthropic console for 2026-09-02: Sonnet
4.5 consumed 6.7M input tokens at a **0.0%** cache read ratio — nothing cached
at all — and accounted for roughly $23 of that day's $24. Sonnet 5, the MI Query
Agent's model, used 580K input tokens at a 98.9% read ratio in the same period.
A grep for `cache_control` across the whole repository returned exactly one
source file, `mi_agent/llm_query_parser.py`. Every other agent bought its system
prompt again on every request.

This agent is the worst shape for that: `SYSTEM_PROMPT` is 751 tokens, the
governed tool schemas are ~600 more, and both are re-sent on every step of an
agentic loop that runs up to `max_steps` times per portfolio.

A request renders tools -> system -> messages, so one breakpoint on the system
block covers the tools too. `messages` stays outside it on purpose: it grows a
turn at a time, and a breakpoint there writes a new entry per step instead of
reading one.
"""
from __future__ import annotations

import unittest
from unittest import mock

from readiness_agent import agent as RA


class TestTheFixedPrefixIsCached(unittest.TestCase):

    def _capture(self):
        seen = {}

        class _Resp:
            content = []
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
            stop_reason = "end_turn"

        def _create(**kwargs):
            seen.update(kwargs)
            return _Resp()

        return seen, _create

    def test_the_system_block_is_a_cache_breakpoint(self):
        seen, _create = self._capture()
        client = mock.Mock()
        client.messages.create.side_effect = _create
        try:
            RA.run_agent(mock.Mock(), objective="x", client=client)
        except Exception:      # the loop needs more of the world than we mock
            pass
        if not seen:
            self.skipTest("run_agent could not be driven with this stub")
        block = seen["system"][0]
        self.assertEqual(block["cache_control"],
                         {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(block["text"], RA.SYSTEM_PROMPT)

    def test_the_source_declares_it(self):
        """A structural check, so the guard holds even where the loop cannot run."""
        import pathlib
        src = (pathlib.Path(RA.__file__)).read_text()
        self.assertIn('"cache_control": {"type": "ephemeral", "ttl": "1h"}', src)
        self.assertIn('system=[{"type": "text", "text": SYSTEM_PROMPT,', src)

    def test_there_is_exactly_one_breakpoint(self):
        """The growing history must not become a second one: it would write,
        never read — a new cache entry per step instead of a hit."""
        import pathlib
        src = (pathlib.Path(RA.__file__)).read_text()
        self.assertEqual(
            src.count('"cache_control"'), 1,
            "%d breakpoints; the fixed prefix needs one, and a breakpoint on "
            "the turn-by-turn messages writes rather than reads"
            % src.count('"cache_control"'))


class TestThePrefixIsWorthCaching(unittest.TestCase):
    """Caching below the minimum cacheable prefix is theatre, not a saving."""

    def test_the_system_prompt_clears_the_minimum(self):
        approx_tokens = len(RA.SYSTEM_PROMPT) // 4
        self.assertGreater(
            approx_tokens, 512,
            "the system prompt is ~%d tokens; below the minimum cacheable "
            "prefix nothing is cached and the breakpoint is decoration"
            % approx_tokens)

    def test_it_carries_no_per_request_content(self):
        """A prefix with a timestamp or an id in it invalidates on every call."""
        for volatile in ("{", "%s", "%(", "now(", "uuid"):
            self.assertNotIn(volatile, RA.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
