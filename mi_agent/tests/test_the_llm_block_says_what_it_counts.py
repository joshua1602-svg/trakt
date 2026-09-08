#!/usr/bin/env python3
"""`metadata.llm` counts the PARSER's calls. It should say so.

WHY THIS EXISTS. Diagnosing the region double-bind, the live response carried

    "llm": {"calls": 0, ...}

alongside, further down the same metadata,

    "conceptMerge": {"status": "applied", "model": "claude-opus-5",
                     "usage": {"input_tokens": 18, "output_tokens": 145,
                               "cache_read_input_tokens": 4755}}

A model had run and written a filter onto the spec. `llm.calls = 0` is true of
what it measures — `llm_query_parser`'s own repair loop — and false of the
question anyone actually asks it, which is "did a model touch this answer".
Read as the latter it refuted the correct hypothesis and cost a diagnosis.

The counter is not changed: consumers read it, and the concept-merge arm keeps
its own usage and cost where they belong. It now NAMES ITS SCOPE, so the next
reader does not have to already know.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _empty_llm_meta


def test_the_block_names_what_it_counts():
    meta = _empty_llm_meta("anthropic", None)
    assert meta.get("scope") == "parser"


def test_the_counter_itself_is_unchanged():
    """Additive only — every field a consumer already reads still reads the
    same, so a dashboard totalling parser cost is untouched."""
    meta = _empty_llm_meta("anthropic", "m")
    assert meta["calls"] == 0
    assert meta["provider"] == "anthropic" and meta["model"] == "m"
    assert meta["total_tokens"] == 0
    assert meta["cost_estimate_status"] == "n/a"
