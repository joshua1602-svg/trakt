#!/usr/bin/env python3
"""clause_splitting_phase1/scoring.py

Scores a projected ``Spans`` against a probe's declared expectation.

The instruments assert CLASSIFICATION, not outcome (§4.0a). A probe passes only
when every span it declares matches and every span it does not declare is
empty. A spurious binding — a filter the question never asked for, a grouping
invented from a dimension value — fails the probe just as a missing one does,
because a spurious binding is the defect class this layer exists to retire.

A failure is additionally tagged ``structural`` when the expectation can only
be met by a channel the tree under test does not have (an unresolvable span, or
unrecognised residue). Structural misses are reported separately so a tree is
never credited with a defect it merely cannot express, and never scored as if
it had answered wrongly when it had nowhere to put the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .span_model import EMPTY, SPAN_NAMES, UNRESOLVABLE, Spans

#: Keys a probe may declare under ``expect``.
_EXPECT_KEYS = set(SPAN_NAMES) | {"unresolved_residue"}


@dataclass
class ProbeResult:
    probe_id: str
    tree: str
    question: str
    ok: bool
    mismatches: List[str] = field(default_factory=list)
    structural_misses: List[str] = field(default_factory=list)
    observed: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[str] = None
    right_for_wrong_reason: bool = False

    @property
    def structural_only(self) -> bool:
        """True when every mismatch is a channel the tree does not have."""
        return bool(self.mismatches) and len(self.structural_misses) == len(self.mismatches)


def _norm(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return sorted(str(v) for v in value)


def _expected_state(decl) -> str:
    """A declaration of ``{unresolvable: "..."}`` expects the unresolvable state."""
    if isinstance(decl, dict) and "unresolvable" in decl:
        return UNRESOLVABLE
    return "any"


def _expected_concepts(decl) -> List[str]:
    if isinstance(decl, dict):
        if "unresolvable" in decl:
            return []
        return _norm(decl.get("concepts"))
    return _norm(decl)


def score(probe: Dict[str, Any], spans: Spans, tree: str,
          has_unresolvable_channel: bool = True) -> ProbeResult:
    """Score one projected split against one probe."""
    expect = probe.get("expect") or {}
    unknown = set(expect) - _EXPECT_KEYS
    if unknown:
        raise ValueError("probe %s declares unknown expectation keys: %s"
                         % (probe.get("id"), sorted(unknown)))

    res = ProbeResult(
        probe_id=probe["id"],
        tree=tree,
        question=probe["question"],
        ok=True,
        observed=spans.as_dict(),
        expected=dict(expect),
        shape=probe.get("shape"),
        right_for_wrong_reason=bool(probe.get("right_for_wrong_reason")),
    )

    for name in SPAN_NAMES:
        decl = expect.get(name)
        want_state = _expected_state(decl)
        want = _expected_concepts(decl)
        got = spans.concepts_of(name)
        got_state = spans.state_of(name)

        if want_state == UNRESOLVABLE:
            if got_state != UNRESOLVABLE:
                msg = "%s: expected unresolvable (%r), got state=%s concepts=%s" % (
                    name, decl.get("unresolvable"), got_state, got)
                res.mismatches.append(msg)
                # Structural ONLY when the tree has no unresolvable channel AND
                # put nothing in the span. A tree that bound the span to a
                # concrete concept made a substantive wrong call, not a
                # structural omission, and is not excused by the missing state.
                if not has_unresolvable_channel and not got:
                    res.structural_misses.append(msg)
            continue

        if got != want:
            res.mismatches.append(
                "%s: expected %s, got %s" % (name, want or "<empty>", got or "<empty>"))
            continue

        # Concepts agree; a span declared as filled must not be reported as
        # unresolvable, and vice versa.
        if want and got_state == UNRESOLVABLE:
            res.mismatches.append("%s: concepts match but span is unresolvable" % name)

    want_residue = bool(expect.get("unresolved_residue"))
    got_residue = bool(spans.unresolved_residue)
    if want_residue and not got_residue:
        msg = "unresolved_residue: expected %s, got none" % _norm(expect["unresolved_residue"])
        res.mismatches.append(msg)
        if not has_unresolvable_channel:
            res.structural_misses.append(msg)  # no channel: nowhere to report it
    elif got_residue and not want_residue:
        res.mismatches.append(
            "unresolved_residue: expected none, got %s" % sorted(spans.unresolved_residue))

    res.ok = not res.mismatches
    return res


def tabulate(results_by_tree: Dict[str, List[ProbeResult]]) -> Dict[str, Any]:
    """Headline counts per tree, plus the structural breakdown."""
    out = {}
    for tree, results in results_by_tree.items():
        passed = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]
        structural = [r for r in failed if r.structural_only]
        out[tree] = {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "failed_structural_only": len(structural),
            "failed_substantive": len(failed) - len(structural),
            "pass_rate": (len(passed) / len(results)) if results else 0.0,
            "by_shape": _by_shape(results),
        }
    return out


def _by_shape(results: List[ProbeResult]) -> Dict[str, Dict[str, int]]:
    shapes: Dict[str, Dict[str, int]] = {}
    for r in results:
        b = shapes.setdefault(r.shape or "unclassified",
                              {"total": 0, "passed": 0, "failed": 0})
        b["total"] += 1
        b["passed" if r.ok else "failed"] += 1
    return shapes
