#!/usr/bin/env python3
"""A pipeline case has an age, and so does the person on it.

THE OBSERVATION, from the 100-question atomic perimeter run:

    "What is the average pipeline case age in days?"
        → *"Average Borrower Age: 74 · 10 loans · entire pipeline"*

Seventy-four is a plausible number for a lifetime-mortgage book, which is what
makes it the worst kind of wrong: a reader checking whether the pipeline is
ageing gets the mean age of the borrowers in it, correctly computed, under a
heading close enough to the question to pass. P048 in that bank.

WHY THE SUBSTITUTION, AND WHY IT IS NOT A SYNONYM-PRIORITY BUG.
`prepare_pipeline_mi_dataset` computes `pipeline_case_age_days` and populates it
on every row — it is a governed field of the pipeline contract and has been
since the contract was written. It is simply absent from
`mi_semantics_field_registry.yaml`, which is the parser's only vocabulary. So
the concept had NO NAME, and `youngest_borrower_age` carries the bare synonym
`age`, which was then the closest thing in the registry to the words the reader
wrote.

An absent concept does not produce a refusal. It produces a substitution by
whichever neighbour claims the word. That is the whole mechanism, and it is why
the fix is to give the field its name rather than to take `age` away from
borrower age — which the brief forbids and which would have broken the twenty
age questions that passed.

THE MULTI-WORD RULE IS WHAT KEEPS THEM APART, and it is pre-existing: both
`_detect_metric` and `_measure_hits` try registry MULTI-WORD phrases first,
longest first, before the curated single tokens. So "pipeline case age" (a
registered phrase) beats "age" (a curated token) without either vocabulary
being weakened. Both halves are asserted below, because the separation is a
property of that ordering and not of the new entry alone.

P049 AND P050 ARE PHASE 2 and are asserted here only as far as Phase 1 reaches:
they refused before this change (the facet guard caught the threshold binding to
borrower age) and they must not start answering wrongly because of it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse       # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics        # noqa: E402

_REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_SEMANTICS = load_mi_semantics(str(_REGISTRY))

CASE_AGE = "pipeline_case_age_days"
BORROWER_AGE = "youngest_borrower_age"


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS)
    return spec


class TestTheFieldHasAName(unittest.TestCase):

    def test_the_registry_carries_the_field_the_pipeline_frame_produces(self):
        """The gap itself. `pipeline_prep` has produced this column all along."""
        self.assertIn(CASE_AGE, _SEMANTICS.get("fields") or {})

    def test_the_prepared_pipeline_frame_really_produces_it(self):
        """Asserted from the CONTRACT rather than restated here, so registering
        a field the pipeline does not produce fails at this line."""
        import yaml

        with open(_REPO_ROOT / "config" / "mi" / "pipeline_field_contract.yaml",
                  "r", encoding="utf-8") as fh:
            contract = yaml.safe_load(fh) or {}
        self.assertIn(CASE_AGE, contract.get("pipeline_specific_fields") or {})


class TestTheTwoAgesStayApart(unittest.TestCase):
    """The contrast pairs the brief required, in both directions."""

    CASE = ("What is the average pipeline case age in days?",
            "What is the average case age?",
            "What is the average number of days in pipeline?",
            "How old are the pipeline cases on average?")

    BORROWER = ("What is the average borrower age in the pipeline?",
                "What is the average youngest borrower age?",
                "What is the average customer age?",
                "What is the average age of borrowers?")

    def test_case_age_language_binds_the_case(self):
        for question in self.CASE:
            with self.subTest(question=question):
                self.assertEqual(parse(question).metric, CASE_AGE)

    def test_borrower_age_language_still_binds_the_borrower(self):
        for question in self.BORROWER:
            with self.subTest(question=question):
                self.assertEqual(parse(question).metric, BORROWER_AGE)

    #: The three SLOTS a borrower-age request can land in — measure, grouping
    #: axis, predicate — with the exact binding each had before this change.
    #: Written per-slot because an earlier draft asserted `metric or dimension`
    #: and short-circuited: "balance by age band" has its age in the DIMENSION
    #: while the metric is the balance, and the assertion read the balance.
    UNCHANGED = (
        ("What is the average age?", "metric", BORROWER_AGE),
        ("Show balance by age band", "dimension", "age_bucket"),
        ("How many funded loans have a borrower aged 85 or older?",
         "filter", BORROWER_AGE),
    )

    def test_the_bare_word_age_is_still_the_borrower_s(self):
        """The vocabulary that was NOT weakened. `age` alone, and every slot the
        age theme scored 20/20 on, keep the meaning they had."""
        for question, slot, expected in self.UNCHANGED:
            with self.subTest(question=question, slot=slot):
                spec = parse(question)
                if slot == "metric":
                    self.assertEqual(spec.metric, expected)
                elif slot == "dimension":
                    self.assertEqual(spec.dimension, expected)
                else:
                    self.assertIn(expected, spec.filters or {})
                self.assertNotEqual(spec.metric, CASE_AGE)


class TestTheSeparationIsTheMultiWordRule(unittest.TestCase):
    """Both resolvers, because the parser uses one and the measure-set guard the
    other, and a fix that satisfied only one would drift the moment a question
    named two measures."""

    def test_the_single_measure_resolver_prefers_the_phrase(self):
        from mi_agent.llm_query_parser import _detect_metric

        key, _agg, matched = _detect_metric("average pipeline case age in days",
                                            _SEMANTICS)
        self.assertEqual(key, CASE_AGE)
        self.assertTrue(any("case age" in m for m in matched), matched)

    def test_the_measure_set_resolver_prefers_the_phrase(self):
        from mi_agent.llm_query_parser import _measure_hits

        keys = [h[2] for h in _measure_hits("pipeline case age and balance",
                                            _SEMANTICS)]
        self.assertIn(CASE_AGE, keys)
        self.assertNotIn(BORROWER_AGE, keys)


class TestPhase1DoesNotStartAnsweringPhase2(unittest.TestCase):
    """P049/P050 stated a threshold on case age. Before this change they bound
    it to borrower age and the facet guard refused. Registering the field must
    not turn either into a confident answer on the wrong column — if they answer
    at all, they answer on the case."""

    def test_a_case_age_threshold_never_lands_on_the_borrower(self):
        for question in ("How many pipeline cases are older than 30 days?",
                         "What is the total pipeline amount for cases older "
                         "than 30 days?"):
            with self.subTest(question=question):
                self.assertNotIn(BORROWER_AGE, parse(question).filters or {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
