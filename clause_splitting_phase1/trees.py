#!/usr/bin/env python3
"""clause_splitting_phase1/trees.py

The trees a probe can be run against.

A tree is a callable ``question -> Spans`` plus a declaration of which span
states it can express. Registering the branch tree later is a one-line
addition; nothing about the instruments changes when it appears.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .span_model import Spans  # noqa: E402

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

_cached_semantics: Optional[dict] = None


def load_semantics(overrides: Optional[dict] = None) -> dict:
    """The registry both trees resolve concepts against.

    ``overrides`` maps a field key to extra synonyms, applied to a DEEP COPY.
    The file on disk is never written, so the tagged release candidate stays
    exactly as tagged while the blast-radius test runs.
    """
    global _cached_semantics
    from mi_agent.mi_query_validator import load_mi_semantics
    if _cached_semantics is None:
        _cached_semantics = load_mi_semantics(_SEMANTICS_PATH)
    if not overrides:
        return _cached_semantics
    import copy
    sem = copy.deepcopy(_cached_semantics)
    fields = sem.get("fields", sem)
    for key, extra in overrides.items():
        if key not in fields:
            raise KeyError("blast-radius synonym targets unknown field %r" % key)
        syns = list(fields[key].get("synonyms") or [])
        for term in extra:
            if term not in syns:
                syns.append(term)
        fields[key]["synonyms"] = syns
    return sem


def rc_tree(semantics: Optional[dict] = None) -> Callable[[str], Spans]:
    """The tagged release candidate, projected into spans."""
    from mi_agent.llm_query_parser import _deterministic_parse
    from .rc_projection import project
    sem = semantics if semantics is not None else load_semantics()

    def _run(question: str) -> Spans:
        spec, meta = _deterministic_parse(question, sem)
        spans = project(spec)
        note = (meta or {}).get("note")
        if note:
            spans.notes.append("rc-meta: %s" % note)
        if (meta or {}).get("dimension_substituted"):
            spans.notes.append("rc-meta: dimension_substituted")
        return spans

    return _run


def rules_reference_tree(semantics: Optional[dict] = None) -> Callable[[str], Spans]:
    """The rules file, run by the reference splitter.

    An ANALYSIS TOOL, not the Phase 2 layer. It exists so the rules can be run
    read-only over the corpus. Scoring it on the probes is indicative of what
    the rules can express; it is not a branch result, and it must not be
    reported as one.
    """
    from .reference_splitter import ReferenceSplitter
    from .resolve import resolve
    splitter = ReferenceSplitter()
    sem = semantics if semantics is not None else load_semantics()

    def _run(question: str) -> Spans:
        return resolve(splitter.split(question), sem)

    return _run


_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")

_cached_book = None


def _book():
    """The real funded tape the release candidate grades against.

    Never build_fixture: the bank refuses that fallback because it is what let
    the bank report 251/252 while scoring 125/252 on a real book.
    """
    global _cached_book
    if _cached_book is None:
        from mi_agent.mi_calibration import default_bank_frame
        _cached_book = default_bank_frame(_REAL_TAPE)
    return _cached_book


def rc_workflow_tree(semantics: Optional[dict] = None) -> Callable[[str], Spans]:
    """The release candidate read END TO END, spec plus execution receipt.

    The parse-only `rc` tree cannot see the facet layer, where the release
    candidate does much of its classification, so it understates it. This tree
    is the fair reading and is the one a Phase 2 comparison must use.
    """
    from mi_agent.mi_agent_workflow import run_mi_agent_query
    from .rc_projection import project_with_receipt
    sem = semantics if semantics is not None else load_semantics()

    def _run(question: str) -> Spans:
        df = _book()
        try:
            res = run_mi_agent_query(question, df, sem)
        except Exception as exc:                       # noqa: BLE001
            spans = Spans()
            spans.notes.append("rc-workflow raised: %s" % exc)
            return spans
        spec = res.get("spec_obj")
        spans = project_with_receipt(spec, res.get("execution_receipt"))
        if not res.get("ok"):
            spans.notes.append("rc-workflow: ok=False — %s"
                               % str(res.get("error"))[:160])
        return spans

    return _run


#: name -> (factory, expresses the unresolvable state)
TREES: Dict[str, tuple] = {
    "rc": (rc_tree, False),
    "rc_workflow": (rc_workflow_tree, True),
    "rules_reference": (rules_reference_tree, True),
}


def get_tree(name: str, semantics: Optional[dict] = None):
    if name not in TREES:
        raise KeyError("unknown tree %r; known: %s" % (name, sorted(TREES)))
    factory, has_unresolvable = TREES[name]
    return factory(semantics), has_unresolvable
