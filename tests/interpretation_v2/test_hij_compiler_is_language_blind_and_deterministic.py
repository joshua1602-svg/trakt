"""H, I, J · The compiler never re-reads the question, and is deterministic.

H is the rule the whole replacement architecture depends on. The previous design
failed because deterministic Python was still doing natural-language
understanding; if the compiler re-read the sentence, this one would be the same
design with an extra step.

It is asserted structurally (the module does not import ``re`` and never touches
the question outside provenance) AND behaviourally (the same intent under two
completely different question strings produces an identical plan).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mi_agent.interpretation_v2 import (
    OUTCOME_PLAN,
    CompilerContext,
    DeterministicCompiler,
    IntentProvenance,
    compare_results,
    parse_candidate_intent,
)
from mi_agent.interpretation_v2 import compiler as compiler_module

from .conftest import intent_payload


# --------------------------------------------------------------------------- #
# H · the compiler does not re-read natural language
# --------------------------------------------------------------------------- #

def test_the_compiler_module_does_not_import_a_regex_engine():
    source = Path(inspect.getfile(compiler_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "re" not in imported, (
        "the compiler imported a regex engine — the one thing it must not do")
    assert "regex" not in imported


def test_the_compiler_touches_the_question_only_to_record_it():
    """``provenance.question`` may be COPIED. It may not be read."""
    source = Path(inspect.getfile(compiler_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "question":
            parent_ok = False
            # The only permitted use is `question=intent.provenance.question`
            # inside _provenance, which is an assignment into a keyword.
            for candidate in ast.walk(tree):
                if isinstance(candidate, ast.keyword) and candidate.arg == "question" \
                        and candidate.value is node:
                    parent_ok = True
            if not parent_ok:
                reads.append(ast.dump(node)[:80])
    assert reads == [], f"the compiler reads the question text: {reads}"


def test_the_same_intent_under_two_different_questions_compiles_identically(compiler):
    """The behavioural half of H, and the one that would catch a subtle read."""
    payload = intent_payload(
        operation="breakdown",
        measures=[{"concept": "balance", "statistic": "sum"}],
        dimensions=["product_type"])

    first = parse_candidate_intent(payload, provenance=IntentProvenance(
        question="Show balance by product type.", model_id="a"))
    second = parse_candidate_intent(payload, provenance=IntentProvenance(
        question="Wie hoch ist der Saldo je Produkttyp? 12345 !!! SELECT * "
                 "FROM loans GROUP BY region",
        model_id="b"))

    left = compiler.compile(first)
    right = compiler.compile(second)
    assert left.outcome == right.outcome == OUTCOME_PLAN
    assert left.plan.plan_id == right.plan.plan_id
    # Provenance is the ONLY difference.
    assert left.plan.provenance.question != right.plan.provenance.question
    assert left.plan._authorised_content() == right.plan._authorised_content()


def test_source_span_evidence_does_not_change_the_plan(compiler):
    """Evidence is provenance. Reading it to decide anything would rebuild the
    recogniser cascade inside the compiler."""
    base = intent_payload(measures=[{"concept": "balance"}])
    without = parse_candidate_intent(base)
    with_evidence = parse_candidate_intent(dict(
        base, evidence=[{"claim": "measure", "text": "the total balance"},
                        {"claim": "period", "text": "as things stand"}]))
    assert (compiler.compile(without).plan.plan_id
            == compiler.compile(with_evidence).plan.plan_id)


# --------------------------------------------------------------------------- #
# I · determinism
# --------------------------------------------------------------------------- #

def test_the_same_intent_compiles_identically_every_time(vocabulary):
    payload = intent_payload(
        operation="breakdown",
        measures=[{"concept": "current_ltv", "statistic": "weighted_average"}],
        dimensions=["product_type", "ltv_bucket"],
        filters=[{"concept": "borrower_age", "comparator": "gte", "value": 60}],
        geography={"requested": True, "level": "itl3", "basis": "obligor",
                   "group_by": True})
    intent = parse_candidate_intent(payload)

    ids = set()
    for _ in range(20):
        compiler = DeterministicCompiler(CompilerContext(vocabulary))
        result = compiler.compile(intent)
        assert result.outcome == OUTCOME_PLAN
        ids.add(result.plan.plan_id)
    assert len(ids) == 1


def test_a_refusal_is_deterministic_too(vocabulary):
    intent = parse_candidate_intent(intent_payload(
        measures=[{"concept": "ebitda"}]))
    outcomes = set()
    for _ in range(10):
        result = DeterministicCompiler(CompilerContext(vocabulary)).compile(intent)
        outcomes.add((result.outcome, tuple(result.codes())))
    assert len(outcomes) == 1


def test_a_different_governed_context_may_legitimately_differ(vocabulary):
    """Determinism is per context, not across them. A book that lacks a field
    must not produce the same plan as one that has it."""
    intent = parse_candidate_intent(intent_payload(
        measures=[{"concept": "indexed_ltv", "statistic": "weighted_average"}]))
    full = DeterministicCompiler(CompilerContext(vocabulary)).compile(intent)
    scoped = DeterministicCompiler(CompilerContext(
        vocabulary, available_fields=["current_outstanding_balance"])).compile(intent)
    assert full.outcome == OUTCOME_PLAN
    assert scoped.outcome != OUTCOME_PLAN


# --------------------------------------------------------------------------- #
# J · paraphrase equivalence
# --------------------------------------------------------------------------- #

def test_paraphrases_with_the_same_meaning_compile_to_the_same_plan(compiler):
    """Three intents that differ only in the ORDER and PROVENANCE the model
    happened to produce must be one authorised plan, not three."""
    variants = [
        intent_payload(
            operation="breakdown",
            measures=[{"concept": "balance", "statistic": "sum"}],
            dimensions=["ltv_bucket", "product_type"],
            evidence=[{"claim": "measure", "text": "balance"}]),
        intent_payload(
            operation="breakdown",
            measures=[{"concept": "balance", "statistic": "sum"}],
            dimensions=["product_type", "ltv_bucket"],
            evidence=[{"claim": "measure", "text": "outstanding balance"}]),
        intent_payload(
            operation="breakdown",
            measures=[{"concept": "balance"}],
            dimensions=["product_type", "ltv_bucket"]),
    ]
    results = []
    for index, payload in enumerate(variants):
        intent = parse_candidate_intent(payload, provenance=IntentProvenance(
            question=f"paraphrase {index}"))
        results.append((f"V{index}", compiler.compile(intent)))

    report = compare_results("canonical", results)
    assert report.invariant, report.divergent_slots


def test_a_genuine_difference_in_meaning_is_reported_as_divergence(compiler):
    left = parse_candidate_intent(intent_payload(
        measures=[{"concept": "balance", "statistic": "sum"}]))
    right = parse_candidate_intent(intent_payload(
        measures=[{"concept": "balance", "statistic": "average"}]))
    report = compare_results("canonical", [("A", compiler.compile(left)),
                                           ("B", compiler.compile(right))])
    assert not report.invariant
    assert "outputs" in report.divergent_slots


def test_two_paraphrases_that_both_refuse_for_the_same_reason_are_invariant(compiler):
    """A refusal is a governed answer. Consistently refusing is consistency."""
    payload = intent_payload(measures=[{"concept": "ebitda"}])
    results = [(qid, compiler.compile(parse_candidate_intent(
        payload, provenance=IntentProvenance(question=qid))))
        for qid in ("A", "B", "C")]
    report = compare_results("canonical", results)
    assert report.invariant
