#!/usr/bin/env python3
"""Does routing a question through a QueryPlan change the answer? Corpus-wide.

THE PRE-D4 QUESTION, asked the only way that can answer it. Every other layer
compares the product with itself at one point: the spec named the right field,
the plan reconciled against the execution, the receipt recorded the filter. None
of them can say whether the ANSWER a reader receives is the same one they would
have received before the lift — and that is the whole of what "switching
QueryPlan on" risks.

So this replays the entire corpus down two paths FROM ONE PARSE, executes both
over a real book, and compares the executed frames cell for cell:

    question ──► _deterministic_parse ──► spec A ──► execute_mi_query ──► frame A
                                            │
                                            ├──► compiled_spec_for ──► frame B1
                                            └──► plan_from_spec
                                                 └► execute_query_plan ─► frame B2

ONE PARSE, deliberately. Parsing twice would compare two questions and call the
result a lift. The estate has already paid for a chat path that ROUTED on one
spec and EXECUTED another.

AN ERROR IS AN OUTCOME. A path that raises where the other answers is the most
serious difference there is — it is how a refusal becomes a number — so a raise
is compared as a value, not skipped.

WHAT B1 AND B2 ARE, and they are not the same thing.

  B1 is what production does today: `compiled_spec_for`, at the single parse
  site. The plan becomes the semantic contract and the reader's PRESENTATION is
  carried across unchanged. This is the invariant that must hold exactly.

  B2 is `execute_query_plan`, which compiles a plan that has no originating
  spec and therefore derives presentation from the analysis — no axis is a
  summary, one is a bar, two are a matrix. It has no production caller today;
  it is how the multi-output composition capability will execute.

They differ in exactly one circumstance, and this file states it as a rule
rather than a list: where the ORIGINATING spec names a chart it has no axis to
draw, path A raises (the executor rejects the shape) and B2 answers the summary
the plan actually describes. That is presentation the plan does not carry, and
it is the reason B1 exists.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mi_agent.execution_receipt as receipt                            # noqa: E402
from mi_agent.llm_query_parser import _deterministic_parse              # noqa: E402
from mi_agent.mi_query_executor import execute_mi_query                 # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics               # noqa: E402
from mi_agent.query_plan_adapter import (                               # noqa: E402
    compiled_spec_for, plan_from_spec)
from mi_agent.query_plan_execution import execute_query_plan            # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth              # noqa: E402
from mi_agent.tests import semantic_census as census_mod                # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

#: A book the executor can actually run: canonical column names, governed
#: values, pre-materialised buckets, and deterministic. The ESMA-schema
#: portfolio the census resolves against does not carry the canonical measure
#: columns, so every execution over it would raise on both paths and the
#: comparison would prove nothing.
_BOOK = truth.canonical_book()
_COLUMNS = receipt.book_columns(_BOOK)
_VALUES = receipt.book_values(_BOOK, _SEMANTICS)


def _signature(result) -> str:
    """The executed frame, canonicalised: sorted columns, sorted rows, rounded.

    Column and row ORDER are presentation. Comparing them would report a
    difference where none of the figures moved, and the point of this replay is
    the figures.
    """
    frame = getattr(result, "data", None)
    if frame is None:
        return "NO_FRAME"
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    frame = frame.reindex(sorted(frame.columns), axis=1)
    if len(frame.columns):
        frame = frame.sort_values(list(frame.columns),
                                  kind="mergesort").reset_index(drop=True)
    return frame.round(6).to_json(orient="records")


def _outcome(call):
    """``("ok", signature)`` or ``("err", "TypeName: message")``."""
    try:
        return ("ok", call())
    except Exception as exc:  # noqa: BLE001 - a raise IS the outcome here
        return ("err", f"{type(exc).__name__}: {exc}")


def _names_a_chart_it_cannot_draw(spec) -> bool:
    """The one circumstance in which B2 may legitimately differ from A."""
    axes = list(spec.dimensions or []) or ([spec.dimension] if spec.dimension else [])
    return spec.intent == "chart" and not axes and not spec.x


class _Replay:
    """One pass over the corpus, shared by every test in this file."""

    rows = None

    @classmethod
    def build(cls):
        if cls.rows is not None:
            return cls.rows
        rows = []
        for question in census_mod.corpus():
            try:
                spec, _meta = _deterministic_parse(
                    question, _SEMANTICS, available_columns=_COLUMNS,
                    available_values=_VALUES)
            except Exception:  # noqa: BLE001 - a parse failure is not this test's
                continue
            if spec is None:
                continue
            try:
                plan = plan_from_spec(spec)
            except Exception:  # noqa: BLE001
                plan = None
            if plan is None:
                continue                      # not liftable; nothing to compare
            lifted = compiled_spec_for(spec)

            def run_plan(plan=plan):
                envelope = execute_query_plan(plan, _BOOK, _SEMANTICS,
                                              validate=False)
                seen = {id(o.execution_ref): o.execution_ref
                        for o in envelope.outputs}
                return "|".join(sorted(_signature(r) for r in seen.values()))

            rows.append({
                "question": question,
                "spec": spec,
                "a": _outcome(lambda spec=spec: _signature(
                    execute_mi_query(spec, _BOOK, _SEMANTICS, validate=False))),
                "b1": (_outcome(lambda s=lifted: _signature(
                    execute_mi_query(s, _BOOK, _SEMANTICS, validate=False)))
                       if lifted is not None else None),
                "b2": _outcome(run_plan),
            })
        cls.rows = rows
        return rows


class TestTheLiftChangesNoAnswer(unittest.TestCase):
    """B1 — what production does today. Nothing here may differ."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _Replay.build()

    def test_the_replay_actually_covered_the_corpus(self):
        """A green comparison over an empty set is the failure mode this test
        exists to make impossible."""
        self.assertGreater(len(self.rows), 400,
                           "the replay compared almost nothing")
        answered = [r for r in self.rows if r["a"][0] == "ok"]
        self.assertGreater(len(answered), 200,
                           "almost every execution raised, so the comparison is "
                           "between error messages and proves little")

    def test_every_liftable_question_executes_identically(self):
        differences = [
            (r["question"], r["a"], r["b1"]) for r in self.rows
            if r["b1"] is not None and r["a"] != r["b1"]]
        self.assertEqual(
            differences, [],
            "routing through a QueryPlan changed what executed:\n"
            + "\n".join(f"  {q}\n    A : {a[0]} {str(a[1])[:200]}\n"
                        f"    B1: {b[0]} {str(b[1])[:200]}"
                        for q, a, b in differences))

    def test_every_liftable_spec_compiles(self):
        """A spec a plan can carry must also compile back to one execution. A
        `None` here is a lift that silently declined after claiming it could."""
        declined = [r["question"] for r in self.rows if r["b1"] is None]
        self.assertEqual(declined, [])


class TestThePlanExecutionPathDiffersOnlyOnPresentation(unittest.TestCase):
    """B2 — stated as a rule, so a NEW kind of divergence fails here.

    `execute_query_plan` compiles a plan that has no originating spec, so it
    derives presentation from the analysis. Where the originating spec named a
    chart it has no axis to draw, the executor rejects that shape on path A and
    the plan path answers the summary the plan describes. That single
    circumstance is allowed; anything else is a divergence between two
    representations of the same request.
    """

    @classmethod
    def setUpClass(cls):
        cls.rows = _Replay.build()

    def test_no_divergence_beyond_an_undrawable_chart(self):
        unexplained = [
            (r["question"], r["a"], r["b2"]) for r in self.rows
            if r["a"] != r["b2"] and not _names_a_chart_it_cannot_draw(r["spec"])]
        self.assertEqual(
            unexplained, [],
            "the plan execution path answered differently for a reason that is "
            "not presentation:\n"
            + "\n".join(f"  {q}\n    A : {a[0]} {str(a[1])[:200]}\n"
                        f"    B2: {b[0]} {str(b[1])[:200]}"
                        for q, a, b in unexplained))

    def test_the_known_divergence_is_still_reachable(self):
        """The rule above is only meaningful while the corpus still contains a
        question that exercises it. If this stops firing the rule is untested,
        not satisfied."""
        exercised = [r for r in self.rows
                     if _names_a_chart_it_cannot_draw(r["spec"])
                     and r["a"] != r["b2"]]
        self.assertGreater(len(exercised), 0,
                           "no corpus question exercises the presentation "
                           "divergence any more; the rule above is now unproven")


if __name__ == "__main__":
    unittest.main()
