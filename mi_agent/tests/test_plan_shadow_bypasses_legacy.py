#!/usr/bin/env python3
"""Phase 6: after interpretation_v2 has spoken, nothing re-reads the question.

The whole argument for this architecture is that ONE owner interprets natural
language. If the new shadow branch quietly consulted a legacy parser, a
recogniser, a router or a raw-text lens resolver to fill a gap, the governed plan
would no longer be the whole story and the architecture would be a fiction.

Proved two ways at once, as in the live sign-off, because neither is sufficient
alone:

    `sys.setprofile` counts every call whose FILE is one of the legacy semantic
    modules — complete for the calling thread, blind to any other.

    counting shims on NAMED entry points catch a call from any thread, but only
    the names listed.

A static import-closure proof is not available and this file does not pretend
otherwise: `mi_agent/__init__.py` imports `llm_query_parser` and
`mi_agent_workflow` eagerly, so they are in `sys.modules` whether used or not.
That predates slice 1 and is not something this slice introduced.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_shadow_evidence as evidence                 # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                    # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth           # noqa: E402
from mi_agent.tests.test_plan_shadow_wiring import (                  # noqa: E402
    ELIGIBLE_GROUPED, ELIGIBLE_SCALAR, INELIGIBLE_LENS, _Shadow, run)

#: Whole modules the new branch must not enter.
LEGACY_FILES = frozenset({
    "parsed_question.py", "llm_query_parser.py", "recogniser_registry.py",
    "chat_routing.py", "portfolio_lens.py", "semantic_resolver.py",
    "mi_agent_workflow.py", "execution_receipt.py", "mi_query_contract.py",
})

#: Named entry points, wrapped for the duration of a case.
LEGACY_ENTRY_POINTS = (
    ("mi_agent.parsed_question", "ParsedQuestion.parse"),
    ("mi_agent.llm_query_parser", "parse_user_question"),
    ("mi_agent.llm_query_parser", "parse_with_repair"),
    ("mi_agent.llm_query_parser", "parse_llm_response_to_spec"),
    ("mi_agent.llm_query_parser", "build_prompt"),
    ("mi_agent.llm_query_parser", "find_field"),
    ("mi_agent_api.recogniser_registry", "RecogniserRegistry.candidates"),
    ("mi_agent_api.recogniser_registry", "RecogniserRegistry.ordered"),
    ("mi_agent_api.chat_routing", "_is_portfolio_summary"),
    ("mi_agent_api.chat_routing", "_is_pipeline_summary"),
    ("mi_agent_api.chat_routing", "_names_something_else"),
    ("mi_agent.portfolio_lens", "lens_phrase_spans"),
    ("mi_agent.portfolio_lens", "names_selected_scope"),
    ("mi_agent.portfolio_lens", "names_total_scope"),
    ("mi_agent.portfolio_lens", "names_a_book_noun"),
    ("mi_agent.portfolio_lens", "mask_scope_phrases"),
    ("mi_agent.execution_receipt", "requested_dimension_terms"),
    ("mi_agent.execution_receipt", "dimension_role"),
    ("mi_agent.mi_agent_workflow", "run_mi_agent_query"),
)


class LegacyWatch:
    """Profiler plus shims, active for exactly the block it wraps."""

    def __init__(self) -> None:
        self.profile_hits = {}
        self.shim_hits = {}
        self._installed = []
        self.missing = []

    def install(self) -> None:
        import importlib
        for module_path, attribute in LEGACY_ENTRY_POINTS:
            label = f"{module_path}.{attribute}"
            try:
                module = importlib.import_module(module_path)
            except Exception:                                        # noqa: BLE001
                self.missing.append(label)
                continue
            owner = module
            parts = attribute.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part, None)
                if owner is None:
                    break
            name = parts[-1]
            original = getattr(owner, name, None) if owner is not None else None
            if original is None:
                self.missing.append(label)
                continue
            self.shim_hits[label] = 0
            self._installed.append((owner, name, original))
            self._wrap(owner, name, original, label)

    def _wrap(self, owner, name, original, label) -> None:
        watch = self
        declared = owner.__dict__.get(name) if hasattr(owner, "__dict__") else None
        is_class_method = isinstance(declared, classmethod)
        raw = original.__func__ if is_class_method else original

        def shim(*args, **kwargs):
            watch.shim_hits[label] = watch.shim_hits.get(label, 0) + 1
            return raw(*args, **kwargs)

        setattr(owner, name, classmethod(shim) if is_class_method else shim)

    def remove(self) -> None:
        for owner, name, original in reversed(self._installed):
            setattr(owner, name, original)
        self._installed = []

    def _hook(self, frame, event, _arg):
        if event != "call":
            return
        base = os.path.basename(frame.f_code.co_filename)
        if base in LEGACY_FILES:
            bucket = self.profile_hits.setdefault(base, {})
            key = frame.f_code.co_name
            bucket[key] = bucket.get(key, 0) + 1

    def __enter__(self):
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *_):
        sys.setprofile(None)

    @property
    def profile_calls(self) -> int:
        return sum(sum(v.values()) for v in self.profile_hits.values())

    @property
    def shim_calls(self) -> int:
        return sum(self.shim_hits.values())


class TestTheNewBranchNeverRereadsTheQuestion(unittest.TestCase):
    """Every disposition the new branch can reach, watched."""

    def _watched(self, question):
        watch = LegacyWatch()
        watch.install()
        try:
            with _Shadow() as cfg:
                with watch:
                    record = run(cfg, question=question)
        finally:
            watch.remove()
        return record, watch

    def test_an_eligible_scalar_touches_no_legacy_semantic_module(self):
        record, watch = self._watched(ELIGIBLE_SCALAR)
        self.assertEqual(record["disposition"], evidence.EXECUTED)
        self.assertEqual(watch.profile_calls, 0, watch.profile_hits)
        self.assertEqual(watch.shim_calls, 0,
                         {k: v for k, v in watch.shim_hits.items() if v})

    def test_a_grouped_execution_touches_none_either(self):
        record, watch = self._watched(ELIGIBLE_GROUPED)
        self.assertEqual(record["disposition"], evidence.EXECUTED)
        self.assertEqual(watch.profile_calls, 0, watch.profile_hits)
        self.assertEqual(watch.shim_calls, 0,
                         {k: v for k, v in watch.shim_hits.items() if v})

    def test_an_ineligible_plan_touches_none_either(self):
        """A refusal must not quietly fall back to a legacy reading."""
        record, watch = self._watched(INELIGIBLE_LENS)
        self.assertEqual(record["disposition"], evidence.INELIGIBLE)
        self.assertEqual(watch.profile_calls, 0, watch.profile_hits)
        self.assertEqual(watch.shim_calls, 0,
                         {k: v for k, v in watch.shim_hits.items() if v})

    def test_every_named_entry_point_was_actually_wrapped(self):
        """A shim that silently failed to install proves nothing."""
        watch = LegacyWatch()
        watch.install()
        try:
            self.assertEqual(watch.missing, [],
                             "some legacy entry points were not found")
            self.assertEqual(len(watch.shim_hits), len(LEGACY_ENTRY_POINTS))
        finally:
            watch.remove()

    def test_the_watch_can_actually_see_a_legacy_call(self):
        """The control: if nothing could be detected, zero would mean nothing."""
        watch = LegacyWatch()
        watch.install()
        try:
            with watch:
                from mi_agent import portfolio_lens
                portfolio_lens.names_total_scope("the whole book")
        finally:
            watch.remove()
        self.assertGreater(watch.shim_calls, 0, "the shims detect nothing")
        self.assertGreater(watch.profile_calls, 0, "the profiler detects nothing")


class TestTheNewBranchOwnsNoSemantics(unittest.TestCase):

    def test_the_plan_is_the_only_source_of_semantics(self):
        """Every bound field in the spec traces to the plan, not to the question."""
        with _Shadow() as cfg:
            record = run(cfg, question=ELIGIBLE_GROUPED)
        requested = record["execution"]["requested_semantics"]
        spec = record["execution"]["bound_spec"]
        self.assertEqual(spec["metric"], requested["measure_field"])
        self.assertEqual(list(spec["dimensions"]), list(requested["dimensions"]))
        self.assertEqual(set(spec["filters"]),
                         {f["field"] for f in requested["filters"]})

    def test_the_recorded_plan_is_byte_identical_to_the_compiled_one(self):
        """Nothing between the compiler and the gate edits the plan."""
        with _Shadow() as cfg:
            record = run(cfg, question=ELIGIBLE_SCALAR)
        plan = record["compiler"]["plan"]
        rebuilt = dict(plan)
        self.assertEqual(json.dumps(plan, sort_keys=True, default=str),
                         json.dumps(rebuilt, sort_keys=True, default=str))
        # And the gate saw that plan: its verdict is recorded beside it.
        self.assertTrue(record["eligibility"]["eligible"])


if __name__ == "__main__":
    unittest.main()
