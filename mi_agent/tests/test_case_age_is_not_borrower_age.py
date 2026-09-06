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

    #: "How old are the pipeline cases on average?" is deliberately NOT here.
    #: It says the same thing with an adjective and a copula rather than a noun
    #: phrase, and no measure vocabulary in this parser reads that construction
    #: for ANY field — "how old are the borrowers on average?" does not resolve
    #: either. Registering a field cannot fix a grammar it does not have, and
    #: adding a synonym to fake it would be patching a question string.
    CASE = ("What is the average pipeline case age in days?",
            "What is the average case age?",
            "What is the average number of days in pipeline?")

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


class TestPhase2BindingIsNowRight(unittest.TestCase):
    """P049/P050 — "how many pipeline cases are OLDER THAN 30 DAYS" — are FIXED.

    THIS CLASS USED TO ASSERT THE OPPOSITE, and it changed because its own guard
    fired. It recorded that Phase 1 had not fixed the predicate seam: the
    threshold bound to `youngest_borrower_age`, because the comparative "older"
    resolved through the age reader and the clause carried no "case age" phrase.
    Both questions failed closed, and the docstring said:

        "If a later change makes either of them ANSWER, this test fails — which
         is the point. Answering is only safe once the binding is right."

    A later change made them answer, this test failed, and the binding was
    checked rather than the assertion relaxed. TWO repairs meet here:

      * the PREDICATE'S subject now resolves through the span-role and unit
        owners, so the bound binds `pipeline_case_age_days` — the case's age —
        and not the borrower's;
      * a threshold every row satisfies is no longer reported as lost, so the
        correct answer is no longer refused for having changed no row count.

    VERIFIED INDEPENDENTLY, not taken from the product. Every case in the
    fixture has a KFI Submitted Date of 2026-05-08 against a reporting date of
    2026-06-12 — 35 days — so all ten are older than thirty, and "10 loans" is
    the arithmetically correct answer over a population the filter genuinely
    selected.

    The guarantee this class carries is unchanged in spirit: answering is only
    safe once the binding is right. It now asserts that the binding IS right,
    and would fail again if the field it binds ever moved back.
    """

    QUESTIONS = ("How many pipeline cases are older than 30 days?",
                 "What is the total pipeline amount for cases older than 30 days?")

    #: The case's own age field. The whole defect was that this used to be
    #: `youngest_borrower_age` — the BORROWER's age — for a question about how
    #: long a case has been sitting.
    CASE_AGE_FIELD = "pipeline_case_age_days"

    def test_the_threshold_binds_the_case_age_not_the_borrower_age(self):
        from mi_agent_api.tests.test_stage_movement_query import ask

        for question in self.QUESTIONS:
            with self.subTest(question=question):
                envelope = ask(question)
                filters = (envelope.get("spec") or {}).get("filters") or {}
                self.assertIn(self.CASE_AGE_FIELD, filters,
                              "the case-age threshold is not bound")
                self.assertNotIn(BORROWER_AGE, filters,
                                 "the borrower's age was bound for a question "
                                 "about the case's age")
                self.assertEqual(filters[self.CASE_AGE_FIELD].get("value"), 30.0)

    def test_the_questions_are_answered_over_the_selected_population(self):
        """Ten of ten, because every case in the fixture is 35 days old — a
        filter that keeps every row is still a filter that ran."""
        from mi_agent_api.tests.test_stage_movement_query import ask

        for question in self.QUESTIONS:
            with self.subTest(question=question):
                envelope = ask(question)
                self.assertTrue(envelope.get("ok"),
                                f"refused: {envelope.get('answer')!r}")

    def test_the_population_is_the_one_an_independent_count_gives(self):
        """The oracle for this fixture is its own dates, read here rather than
        asked of the product."""
        import pandas as pd

        frame = pd.read_csv(
            _REPO_ROOT / "tests" / "fixtures" / "pipeline_transition_2w"
            / "2026-06-12" / "M2L_KFI_and_Pipeline_2026_06_12.csv",
            low_memory=False)
        age = (pd.Timestamp("2026-06-12")
               - pd.to_datetime(frame["KFI Submitted Date"],
                                errors="coerce")).dt.days
        self.assertEqual(int((age > 30).sum()), len(frame),
                         "the fixture no longer has every case older than 30 "
                         "days, so the expectation below must be recomputed")

    def test_the_measure_half_of_the_family_is_fixed(self):
        """P048, the one that was answering wrongly, on the same fixture."""
        from mi_agent_api.tests.test_stage_movement_query import ask

        answer = ask("What is the average pipeline case age in days?").get("answer") or ""
        self.assertIn("Pipeline Case Age", answer)
        self.assertNotIn("Borrower Age", answer)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
