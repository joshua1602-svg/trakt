#!/usr/bin/env python3
"""A question that names the pipeline-stage dimension is about the pipeline.

THE DEFECT, from the live 115-question bank. `pipeline_stage_movement` answers
49 of 49. Six questions asking for exactly that were refused:

    What was the largest stage transition?     'Pipeline Stage' is not
    What stage had the most withdrawals?        available in this dataset
    Which stage had the most movement?
    Compare stage movement with the prior period.
    Give me the stage movement summary.
    How did cases move through the funnel?

The refusal is true and three layers downstream of the mistake. THE FIRST WRONG
DECISION is dataset selection: `workspace.resolve_dataset` sent all six to the
FUNDED book, which carries no `pipeline_stage` column, so field binding and
execution were correct to refuse a column that genuinely is not there.

WHY IT SENT THEM THERE. Its pipeline vocabulary was a hand-written triple of
stage VALUES — ("kfi", "application", "offer") — and not one of the six names a
value. They name the DIMENSION: stage, funnel. A question about the axis was
invisible to a rule that only knew three of its points.

THE OWNER. The registry already says which terms name that dimension:
`pipeline_stage.synonyms`. Dataset selection now reads it instead of keeping a
second, smaller copy — the same defect this whole sprint has been closing, one
concept with two vocabularies. Nothing is invented here: a term that moves a
question is a term the registry already binds to `pipeline_stage`, and adding
one is a governed registry change, not a code change.

MEASURED on the 882-question corpus: 3 questions move (0.34%), all three
genuinely about pipeline stages. Every funded question that answers today stays
funded — "Show movement by region." above all, which the word "movement" would
have taken.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import workspace as W

#: The six the bank refused, and the capability that already answers them.
STAGE_QUESTIONS = (
    "What was the largest stage transition?",
    "What stage had the most withdrawals?",
    "Which stage had the most movement?",
    "Compare stage movement with the prior period.",
    "Give me the stage movement summary.",
    "How did cases move through the funnel?",
)

#: Funded questions answering today. Every one must stay funded.
FUNDED_QUESTIONS = (
    "Show movement by region.",
    "Show balance movement by portfolio.",
    "What is funded balance movement?",
    "Why did funded balance increase?",
    "Show balance by region.",
    "How has average LTV changed since last month?",
    "Show the weighted average LTV for the Direct book",
    "What is the total balance for loans with borrowers older than 80?",
)


class TestTheStageDimensionSelectsThePipeline(unittest.TestCase):

    def test_every_stage_question_reaches_the_pipeline(self):
        for question in STAGE_QUESTIONS:
            with self.subTest(question=question):
                self.assertEqual(W.resolve_dataset(question), "pipeline")

    def test_no_funded_question_is_taken_with_them(self):
        for question in FUNDED_QUESTIONS:
            with self.subTest(question=question):
                self.assertEqual(W.resolve_dataset(question), "funded")

    def test_a_question_naming_the_funded_book_outright_still_wins(self):
        """Precedence is unchanged: a view the question NAMES beats this rule."""
        self.assertEqual(W.resolve_dataset("expected funded by stage"), "funded")
        self.assertEqual(W.resolve_dataset("Forecast the stage mix next month"),
                         "forecast")


class TestTheVocabularyIsTheRegistrysOwn(unittest.TestCase):
    """Not a second list. The registry owns which terms name a dimension."""

    def _registry_synonyms(self):
        import yaml

        with open(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml",
                  "r", encoding="utf-8") as fh:
            registry = yaml.safe_load(fh)
        entry = (registry.get("fields") or {}).get("pipeline_stage") or {}
        return set(entry.get("synonyms") or ())

    def test_the_stage_terms_come_from_the_registry(self):
        self.assertTrue(
            self._registry_synonyms() <= set(W.pipeline_dataset_terms()),
            "dataset selection keeps stage terms the registry does not have")

    def test_the_registry_binds_every_term_that_moves_a_question(self):
        """The other direction, which is the one that matters: a term here that
        the registry does not bind to `pipeline_stage` would be an invented
        vocabulary."""
        governed = self._registry_synonyms() | set(W.PIPELINE_ARTEFACTS)
        extra = set(W.pipeline_dataset_terms()) - governed
        self.assertEqual(extra, set(), "invented terms: %s" % sorted(extra))


class TestTheCorpusBarelyMoves(unittest.TestCase):
    """The standard this module's own docstring sets: measure the movement."""

    def _corpus(self):
        path = _REPO_ROOT / "question_interpretation" / "stage2_corpus.json"
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        return sorted({r.get("question") for r in rows if r.get("question")})

    def test_the_rule_moves_a_handful_and_they_are_all_stage_questions(self):
        moved = [q for q in self._corpus()
                 if W.resolve_dataset(q) == "pipeline"
                 and not any(w in q.lower() for w in ("pipeline", "kfi",
                                                      "offer", "application"))]
        self.assertLessEqual(len(moved), 8, moved)
        for question in moved:
            with self.subTest(question=question):
                self.assertRegex(question.lower(), r"stage|funnel")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
