"""No question from any bank is in what the model sees (P0 design §25).

Owner direction, 2026-09-30: "you must not make local / tactical fixes just to
pass the 135 bank. These MUST be supportive of other natural language
variants." A definition that quotes a bank question teaches the model that
SENTENCE; the next reader who words it differently gets whatever the
definition said before. So the model's view — the rules, the orientation, the
governed catalogue with every definition, and the intent tool — states
meanings, with term-level synonyms, and never a question a bank scores.

A bank entry of four words or more is a question; shorter entries ("balance by
region", "weighted average LTV") are the names of governed concepts, which the
catalogue must name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from mi_agent.interpretation_v2 import opus_interpreter
from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary

_ROOT = Path(__file__).resolve().parents[2]
_BANKS = sorted({
    *(_ROOT / "mi_agent/interpretation_v2/banks").glob("*.yaml"),
    *(_ROOT / "config/mi/golden_questions").glob("*.yaml"),
    *(_ROOT / "migration_phase0").glob("*BANK*.yaml"),
    *(_ROOT / "question_interpretation").glob("*bank*.yaml"),
    *(_ROOT / "tests/fixtures").rglob("*BANK*.yaml"),
})
_HOLDOUT = _ROOT / "mi_agent/interpretation_v2/banks/holdout_variants_20260930.yaml"
_MIN_WORDS = 4


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9£% ]+", " ", text).split())


def _questions(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("question", "text", "q") and isinstance(value, str):
                yield value
            else:
                yield from _questions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _questions(item)


@pytest.fixture(scope="module")
def models_view() -> str:
    vocabulary = load_governed_vocabulary()
    blocks = opus_interpreter.build_system_blocks(vocabulary)
    text = "\n".join(block["text"] for block in blocks)
    return _norm(text + "\n" + json.dumps(opus_interpreter.build_tool_schema()))


def test_the_banks_are_found():
    names = {p.name for p in _BANKS}
    assert "interpretation_bank_135.yaml" in names
    assert "ere_mi_questions.yaml" in names
    assert _HOLDOUT in _BANKS


@pytest.mark.parametrize("bank", _BANKS, ids=lambda p: p.name)
def test_no_bank_question_is_quoted_to_the_model(bank, models_view):
    quoted = sorted({
        q for q in _questions(yaml.safe_load(bank.read_text(encoding="utf-8")))
        if len(_norm(q).split()) >= _MIN_WORDS and _norm(q) in models_view})
    assert quoted == [], (
        f"{bank.name}: state the meaning these rest on, not the question — "
        f"{quoted}")


#: Words that carry no meaning of their own in a question.
_STOP = frozenset(
    "a an the is are was were what whats which who how much many do does did "
    "of to in on for by and or with from our we us me show give tell please "
    "there it its this that be at as".split())


def _content(text: str):
    return [w for w in _norm(text).split() if w not in _STOP]


@pytest.fixture(scope="module")
def models_clauses():
    """The model's view cut at punctuation and quotes — each clause one
    phrase a definition or rule states — with the governed concepts' own
    names (label, id, aliases) taken out: a catalogue must name its concepts
    whatever a bank asks, and a question that IS a concept's name is not
    quoted by naming it."""
    vocabulary = load_governed_vocabulary()
    blocks = opus_interpreter.build_system_blocks(vocabulary)
    raw = ("\n".join(block["text"] for block in blocks)
           + json.dumps(opus_interpreter.build_tool_schema(), ensure_ascii=False))
    raw = raw.replace("\\'", "'")
    names = sorted({_norm(n) for c in vocabulary.concepts.values()
                    for n in (c.label, c.concept_id, *c.aliases) if _norm(n)},
                   key=len, reverse=True)
    naming = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")
    clauses = re.split(r"(?:(?<![a-z])'|'(?![a-z])|[\"\n.;:()\[\]{}—,])", raw)
    return [naming.sub(" ", _norm(c)) for c in clauses if _norm(c)]


def _restated(clauses):
    buckets = json.loads((_ROOT / "due_diligence/evidence/qb_plan_readback/"
                          "qb_question_buckets.json").read_text())
    clause_words = [set(_content(c)) for c in clauses]
    restated = []
    for qid, row in buckets["questions"].items():
        asked = set(_content(row["question"]))
        if len(asked) < _MIN_WORDS:
            continue
        for words in clause_words:
            if (len(asked & words) / len(asked) >= 0.85
                    and len(words - asked) <= 2):
                restated.append((qid, row["question"], sorted(words)))
                break
    return restated


def test_no_bank_question_is_restated_to_the_model(models_clauses):
    """A question reworded by a word or two is still the question. For every
    bank question with four meaning-bearing words or more, no phrase the model
    is shown carries 85% of them with at most two others."""
    assert _restated(models_clauses) == []


def test_the_restatement_guard_catches_what_2_14_0_said(models_clauses):
    """The wordings 2.15.0 replaced, each put back into the view, are caught."""
    for old in ("how much of the forecast comes from the funded book",
                "milestone dates to funding thresholds",
                "project the funded balance over the next N months",
                "how much pipeline has lapsed past its stage window"):
        assert _restated([*models_clauses, old]), old


def test_the_guard_would_catch_a_quoted_question(models_view):
    """The check is not vacuous: a definition quoting a bank question fails."""
    quoted = _norm("When are pipeline cases expected to complete?")
    assert quoted not in models_view
    assert quoted in _norm(f"{models_view} it answers 'when are pipeline "
                           f"cases expected to complete'")


def test_every_holdout_variant_names_a_bank_question_and_its_bucket():
    buckets = json.loads((_ROOT / "due_diligence/evidence/qb_plan_readback/"
                          "qb_question_buckets.json").read_text())
    rows = yaml.safe_load(_HOLDOUT.read_text(encoding="utf-8"))["questions"]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    bank = {_norm(q["question"]) for q in buckets["questions"].values()}
    for row in rows:
        original = buckets["questions"][row["variant_of"]]
        assert row["bucket"] in buckets["buckets"]
        assert row["variant_of"] in buckets["buckets"][row["bucket"]], row["id"]
        # A variant is other words, not the bank sentence again.
        assert _norm(row["question"]) not in bank, row["id"]
        assert _norm(row["question"]) != _norm(original["question"])
