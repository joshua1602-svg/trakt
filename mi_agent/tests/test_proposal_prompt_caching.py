#!/usr/bin/env python3
"""The proposal prompt's cacheable half is cached for as long as it is stable.

WHY THIS FILE EXISTS. The concept-merge arm sends the governed vocabulary as the
system block and the question as the user message. Measured on a book: 2,170
tokens of vocabulary against 13 of question — 99% of every request is a prefix
that does not change until the BOOK changes. It was already cached, at the
default five-minute ephemeral TTL, which an MI dashboard used a few times an
hour misses on nearly every call.

The second test pins something the first outage taught. The cached call is
wrapped in a bare `except Exception` whose comment reads "SDK without cache
support", and on 2026-09-02 every one of 139 questions failed there with a 400
saying the credit balance was too low — then silently retried uncached, failed
identically, and billed twice for nothing. A model that has cached successfully
before cannot suddenly not support caching, so the retry is skipped and the real
error is allowed to surface.
"""
from __future__ import annotations

import unittest
from unittest import mock

from mi_agent import llm_query_parser as LQ

PROMPT = {"system": "vocabulary: a b c", "user": "Question: how many?"}


class _Msg:
    content = [type("T", (), {"type": "text", "text": "[]"})()]
    usage = None


class TestTheStablePrefixIsCachedForAWorkingSession(unittest.TestCase):

    def test_the_system_block_carries_an_explicit_hour_ttl(self):
        seen = {}

        def _create(**kwargs):
            seen.update(kwargs)
            return _Msg()

        with mock.patch.object(LQ, "_sampling_for", return_value={}), \
                mock.patch("anthropic.Anthropic") as _client:
            _client.return_value.messages.create.side_effect = _create
            LQ._call_llm(PROMPT, "claude-opus-5")

        block = seen["system"][0]
        self.assertEqual(block["cache_control"],
                         {"type": "ephemeral", "ttl": "1h"},
                         "the vocabulary fell back to the 5-minute default; a "
                         "dashboard used hourly would miss it every time")
        # The QUESTION must stay out of the cached prefix, or the prefix
        # changes on every request and nothing is ever reused.
        self.assertEqual(seen["messages"][0]["content"], PROMPT["user"])


class TestANonCacheErrorIsNotRetriedUncached(unittest.TestCase):
    """Billing and auth faults must surface, not masquerade as a missing feature."""

    def setUp(self):
        LQ._CACHE_SUPPORTED.discard("claude-opus-5")

    def tearDown(self):
        LQ._CACHE_SUPPORTED.discard("claude-opus-5")

    def test_once_a_model_has_cached_a_later_failure_propagates(self):
        calls = []

        def _create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _Msg()                       # first call: caching works
            raise RuntimeError("credit balance is too low")

        with mock.patch.object(LQ, "_sampling_for", return_value={}), \
                mock.patch("anthropic.Anthropic") as _client:
            _client.return_value.messages.create.side_effect = _create
            LQ._call_llm(PROMPT, "claude-opus-5")           # establishes support
            with self.assertRaises(RuntimeError):
                LQ._call_llm(PROMPT, "claude-opus-5")

        self.assertEqual(len(calls), 2,
                         "the failing request was retried uncached and billed "
                         "twice: %d calls" % len(calls))

    def test_a_genuinely_uncacheable_model_still_falls_back(self):
        """The fallback exists for a reason and must survive the guard."""
        calls = []

        def _create(**kwargs):
            calls.append(kwargs)
            if "cache_control" in str(kwargs.get("system")):
                raise TypeError("unexpected keyword 'cache_control'")
            return _Msg()

        with mock.patch.object(LQ, "_sampling_for", return_value={}), \
                mock.patch("anthropic.Anthropic") as _client:
            _client.return_value.messages.create.side_effect = _create
            text, _usage, cached = LQ._call_llm(PROMPT, "some-old-model")

        self.assertFalse(cached)
        self.assertEqual(len(calls), 2, "the uncached retry did not happen")


if __name__ == "__main__":
    unittest.main()
