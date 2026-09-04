#!/usr/bin/env python3
"""One block says whether a model touched the answer, and what it cost.

WHY. The live London response carried, in its metadata:

    "llm": {"calls": 0, ...}

and, further down the same object:

    "conceptMerge": {"status": "applied", "model": "claude-opus-5",
                     "usage": {"input_tokens": 18, "output_tokens": 145,
                               "cache_read_input_tokens": 4755}}

A model HAD run, and had written a filter onto the spec that made the answer
refuse. `llm.calls = 0` is true of what it measures — the free-form parser's own
repair loop — and false of the question every reader actually asks it. Read as
"no model touched this", it refuted a correct diagnosis and cost a day.

Naming the scope of the old counter is not enough: the two numbers still live in
different places and a reader must know to add them. `modelUsage` states the
whole picture in one place, and the replay probe carries it, so the bank evidence
can never again show a model-free parse for a request a model changed.

Nothing is renamed and nothing is removed: `llm` and `conceptMerge` are
unchanged for every consumer that already reads them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api.mi_service import _model_usage

#: The two blocks exactly as the live London response carried them.
LIVE_PARSER = {"scope": "parser", "provider": "anthropic", "model": None,
               "calls": 0, "input_tokens": 0, "output_tokens": 0,
               "total_tokens": 0, "cache_read_tokens": 0,
               "cache_write_tokens": 0, "estimated_total_cost": 0.0}
LIVE_ARM = {"status": "applied", "model": "claude-opus-5",
            "usage": {"input_tokens": 18, "output_tokens": 145,
                      "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 4755},
            "cost": {"estimated_total_cost": 0.006092}}


def test_the_live_london_shape_reports_a_model_call():
    """The exact case that misled the diagnosis."""
    usage = _model_usage(LIVE_PARSER, LIVE_ARM)
    assert usage["free_form_parser_calls"] == 0
    assert usage["concept_merge_calls"] == 1
    assert usage["total_model_calls"] == 1
    assert usage["models"] == ["claude-opus-5"]


def test_the_tokens_and_cost_are_totalled_across_both_arms():
    usage = _model_usage(LIVE_PARSER, LIVE_ARM)
    assert usage["input_tokens"] == 18
    assert usage["output_tokens"] == 145
    assert usage["cache_read_tokens"] == 4755
    assert usage["cache_write_tokens"] == 0
    assert round(usage["estimated_total_cost"], 6) == 0.006092


def test_a_genuinely_model_free_answer_says_so():
    usage = _model_usage(LIVE_PARSER, None)
    assert usage["total_model_calls"] == 0
    assert usage["models"] == []
    assert usage["estimated_total_cost"] == 0.0


def test_an_arm_that_ran_and_changed_nothing_still_counts_as_a_call():
    """`no_change` means the model was asked and agreed — it ran, and it cost."""
    usage = _model_usage(LIVE_PARSER, {**LIVE_ARM, "status": "no_change"})
    assert usage["concept_merge_calls"] == 1
    assert usage["total_model_calls"] == 1


def test_an_arm_that_never_reached_the_model_is_not_a_call():
    usage = _model_usage(LIVE_PARSER, {"status": "proposal_unavailable",
                                       "detail": "boom"})
    assert usage["concept_merge_calls"] == 0
    assert usage["total_model_calls"] == 0


def test_a_replayed_arm_is_not_a_fresh_model_call():
    usage = _model_usage(LIVE_PARSER, {**LIVE_ARM, "source": "replayed"})
    assert usage["concept_merge_calls"] == 0


def test_the_parser_arm_is_counted_when_it_runs():
    parser = {**LIVE_PARSER, "calls": 2, "model": "claude-opus-5",
              "input_tokens": 100, "output_tokens": 50,
              "estimated_total_cost": 0.01}
    usage = _model_usage(parser, None)
    assert usage["free_form_parser_calls"] == 2
    assert usage["total_model_calls"] == 2
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 50


def test_both_arms_running_are_both_counted_once_each(): 
    parser = {**LIVE_PARSER, "calls": 1, "model": "claude-sonnet-5",
              "input_tokens": 10, "output_tokens": 5,
              "estimated_total_cost": 0.001}
    usage = _model_usage(parser, LIVE_ARM)
    assert usage["total_model_calls"] == 2
    assert sorted(usage["models"]) == ["claude-opus-5", "claude-sonnet-5"]
    assert usage["input_tokens"] == 28
    assert round(usage["estimated_total_cost"], 6) == 0.007092


def test_missing_blocks_never_raise():
    assert _model_usage(None, None)["total_model_calls"] == 0
    assert _model_usage({}, {})["total_model_calls"] == 0


def test_the_probe_carries_it_into_the_bank_evidence():
    from migration_phase0.replay_probe import _META_KEYS

    assert "modelUsage" in _META_KEYS
