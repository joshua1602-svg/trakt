#!/usr/bin/env python3
"""One requested concept resolves to ONE physical field, whoever claims it.

THE DEFECT, from the live 115-question replay (2026-09-04). "What is the funded
balance in the London region" was refused on a book that carries London:

    'Region' is not available in this dataset.
    validation: Canonical column 'canonical_region_reporting' not present
    spec.filters: {"collateral_geography": "London",
                   "canonical_region_reporting": "London"}

Two spellings of ONE concept, both written as filters, one of them naming a
column the execution frame does not have. The executor then refused — correctly,
given the spec it was handed.

The response's own evidence names the writer:

    conceptMerge.findings[]
      {"outcome": "filled_by_model", "slot": "row_predicates",
       "key": "canonical_region_reporting", "detail": "the slot was empty"}

The slot was NOT empty. The deterministic parse had already narrowed on
`collateral_geography` = London, and the two fields both declare
``value_domain: uk_region`` — they are aliases of one concept, not two concepts.
`merge` addresses occupancy by PHYSICAL FIELD, so the second spelling found a
free address and filled it.

    (row_predicates, "collateral_geography")     <- the reader's claim
    (row_predicates, "canonical_region_reporting")  <- a free address

This is why GROUPING worked and FILTERING failed. The dimensions path compared
the same key on both sides and correctly returned
`declined_field_already_placed_in_another_role`; the row-predicate path compared
two different keys for one concept and saw an empty slot.

Nothing here is about geography. The rule is addressed by CONCEPT: any two
fields that declare one ``value_domain`` are one address, so the merge's
existing three rules — agree, decline, never overwrite — apply to a concept the
deterministic side has already claimed under another of its spellings.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation import claim_merge as CM
from question_interpretation.concept_proposal import BoundConcept, ProposedConcept

_REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

READER = "collateral_geography"          # what the deterministic parse bound
ALIAS = "canonical_region_reporting"     # what the model's proposal bound
UNRELATED = "broker_channel"


def _semantics():
    with open(_REGISTRY, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _reader_claim(field=READER, value="London"):
    """The deterministic side: a person's own narrowing."""
    return (CM.SlotValue(CM.SLOT_ROW_PREDICATES, field, value,
                         CM.PROV_EXPLICIT_USER),)


def _model_binds(field=ALIAS, value="London", kind="category_value"):
    return (BoundConcept(ProposedConcept(kind=kind, term=str(value).lower()),
                         field, value, "categorical_spans.value_field"),)


def _keys(result):
    return [s.key for s in result.slots if s.slot == CM.SLOT_ROW_PREDICATES]


class TestOneConceptIsOneAddress(unittest.TestCase):

    def test_an_alias_of_a_claimed_concept_is_not_an_empty_slot(self):
        result = CM.merge(_reader_claim(), _model_binds(),
                          semantics=_semantics())
        self.assertEqual(_keys(result), [READER],
                         "the merge wrote a second spelling of one concept")

    def test_the_agreement_is_reported_rather_than_silently_dropped(self):
        """Same concept, same value: the deterministic claim already says it."""
        result = CM.merge(_reader_claim(), _model_binds(),
                          semantics=_semantics())
        outcomes = [f.outcome for f in result.findings]
        self.assertIn(CM.AGREED, outcomes)
        self.assertNotIn(CM.FILLED_BY_MODEL, outcomes)

    def test_a_disagreeing_alias_is_declined_never_applied_as_a_second_filter(self):
        """"London" already claimed; the model says "Scotland" under the alias.
        Neither side is picked and NOTHING is written — a second key here would
        select the intersection, which is no loans at all."""
        result = CM.merge(_reader_claim(), _model_binds(value="Scotland"),
                          semantics=_semantics())
        self.assertEqual(_keys(result), [READER])
        outcomes = [f.outcome for f in result.findings]
        self.assertTrue(any(o.startswith("declined") for o in outcomes), outcomes)

    def test_a_genuinely_new_concept_is_still_filled(self):
        """The arm's whole purpose survives: an UNRELATED field still fills."""
        result = CM.merge(_reader_claim(), _model_binds(field=UNRELATED,
                                                        value="Alpha"),
                          semantics=_semantics())
        self.assertEqual(sorted(_keys(result)), sorted([READER, UNRELATED]))
        self.assertIn(CM.FILLED_BY_MODEL, [f.outcome for f in result.findings])

    def test_the_same_field_twice_still_behaves_exactly_as_before(self):
        result = CM.merge(_reader_claim(), _model_binds(field=READER),
                          semantics=_semantics())
        self.assertEqual(_keys(result), [READER])
        self.assertIn(CM.AGREED, [f.outcome for f in result.findings])

    def test_with_no_semantics_the_rule_stands_down(self):
        """A caller that cannot say which fields are aliases gets exactly the
        behaviour it had before this parameter existed."""
        result = CM.merge(_reader_claim(), _model_binds())
        self.assertEqual(sorted(_keys(result)), sorted([READER, ALIAS]))


class TestTheRuleIsNotAboutGeography(unittest.TestCase):
    """It is addressed by `value_domain`, so it holds for every alias family
    the registry declares — present and future."""

    def test_the_family_comes_from_the_governed_domain_not_a_list(self):
        from mi_agent.categorical_spans import alias_fields

        sem = _semantics()
        family = set(alias_fields(READER, sem))
        self.assertIn(ALIAS, family)
        self.assertNotIn(UNRELATED, family)

    def test_a_field_with_no_value_domain_has_no_aliases(self):
        from mi_agent.categorical_spans import alias_fields

        self.assertEqual(tuple(alias_fields(UNRELATED, _semantics())), ())

    def test_an_unknown_field_has_no_aliases(self):
        from mi_agent.categorical_spans import alias_fields

        self.assertEqual(tuple(alias_fields("not_a_field", _semantics())), ())


class TestTheInvariant(unittest.TestCase):
    """PHASE 2's acceptance rule, asserted directly."""

    def test_at_most_one_physical_key_per_alias_family(self):
        from mi_agent.categorical_spans import alias_fields

        sem = _semantics()
        result = CM.merge(_reader_claim(), _model_binds(), semantics=sem)
        keys = [k for k in _keys(result) if k]
        for key in keys:
            family = set(alias_fields(key, sem)) | {key}
            self.assertLessEqual(
                len([k for k in keys if k in family]), 1,
                "more than one physical key emitted for one concept: %s" % keys)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestAValueBindsOnlyToAColumnTheFrameHas(unittest.TestCase):
    """THE SECOND HALF OF THE SAME DEFECT.

    `_bind` asserts `available_columns` for a MEASURE and for a DIMENSION —
    both say so in their own comments — and does not for a VALUE. The arm's
    catalogue is built from a SEPARATELY resolved frame
    (`chat_routing._values_for_recognition` → `base_frame_resolver`), not the
    frame the parse and the executor use, so the two need not agree about which
    region columns exist. A value bound against the other frame's catalogue
    reached the spec as a filter on a column this frame does not carry, and the
    executor refused at `_require_column`.

    Availability is the SAME question for all three kinds, so it gets the same
    answer.
    """

    def _vocab(self, columns):
        from question_interpretation import concept_proposal as CP

        return CP.vocabulary(
            _semantics(),
            available_values={ALIAS: {"london": "London"}},
            available_columns=columns)

    def _bind_london(self, columns):
        from question_interpretation import concept_proposal as CP

        proposal = ProposedConcept(kind="category_value", term="london")
        return CP.bind([proposal], self._vocab(columns))

    def test_a_value_naming_an_absent_column_is_rejected_not_bound(self):
        bound, rejected = self._bind_london({READER, "current_outstanding_balance"})
        self.assertEqual(list(bound), [], "bound a value to a column the frame lacks")
        self.assertTrue(rejected)

    def test_a_value_whose_column_is_present_still_binds(self):
        bound, _ = self._bind_london({ALIAS, "current_outstanding_balance"})
        self.assertEqual([b.field for b in bound], [ALIAS])

    def test_with_no_column_context_the_rule_stands_down(self):
        bound, _ = self._bind_london(None)
        self.assertEqual([b.field for b in bound], [ALIAS])


class TestSuppressionIsSafeNotMerelyQuieter(unittest.TestCase):
    """"The arm writes less" is not proof that no wrong answer can appear.

    Suppressing a model fill is only safe when the deterministic contract
    ALREADY REPRESENTS THE SAME SEMANTIC CONCEPT. These assert that, and assert
    the cases where suppression must NOT happen — because a narrowing that
    disappears without an equivalent in its place answers over a broader
    population than the reader asked about, which is the failure mode the whole
    facet ledger exists to prevent.
    """

    def _row_slots(self, result):
        return {s.key: s for s in result.slots
                if s.slot == CM.SLOT_ROW_PREDICATES}

    def test_a_suppressed_fill_leaves_an_equivalent_narrowing_in_place(self):
        """The reader's own claim survives, carrying the same value."""
        result = CM.merge(_reader_claim(), _model_binds(), semantics=_semantics())
        held = self._row_slots(result)
        self.assertIn(READER, held)
        self.assertEqual(held[READER].value, "London")
        self.assertTrue(held[READER].chosen_by_a_person)

    def test_nothing_is_suppressed_when_the_contract_holds_no_such_concept(self):
        """No deterministic claim at all: the fill MUST land, or the narrowing
        the reader asked for is silently gone."""
        result = CM.merge((), _model_binds(), semantics=_semantics())
        self.assertEqual(list(self._row_slots(result)), [ALIAS])
        self.assertIn(CM.FILLED_BY_MODEL, [f.outcome for f in result.findings])

    def test_a_second_concept_on_the_same_field_family_is_not_collapsed(self):
        """DISTINCT predicates must not merge merely because they share a field.
        A threshold and a category on one field are two claims, and the merge
        reports the disagreement rather than dropping either."""
        bound = (BoundConcept(ProposedConcept(kind="category_value", term="scotland"),
                              ALIAS, "Scotland", "categorical_spans.value_field"),)
        result = CM.merge(_reader_claim(), bound, semantics=_semantics())
        held = self._row_slots(result)
        self.assertEqual(held[READER].value, "London")
        outcomes = [f.outcome for f in result.findings]
        self.assertTrue(any(o.startswith("declined") for o in outcomes), outcomes)
        # The disagreement is REPORTED, with both sides named, so nothing is
        # lost silently.
        finding = next(f for f in result.findings if f.outcome.startswith("declined"))
        self.assertEqual(finding.proposed, "Scotland")
        self.assertEqual(finding.deterministic, "London")

    def test_two_unrelated_fields_both_survive(self):
        """The rule is per CONCEPT. Two genuinely different concepts are two
        predicates, and suppressing either would broaden the population."""
        bound = (BoundConcept(ProposedConcept(kind="category_value", term="alpha"),
                              UNRELATED, "Alpha", "categorical_spans.value_field"),)
        result = CM.merge(_reader_claim(), bound, semantics=_semantics())
        self.assertEqual(sorted(self._row_slots(result)),
                         sorted([READER, UNRELATED]))

    def test_no_deterministic_narrowing_is_ever_removed(self):
        """Whatever the model proposes, every predicate the reader's own parse
        recorded is still on the contract afterwards."""
        reader = (CM.SlotValue(CM.SLOT_ROW_PREDICATES, READER, "London",
                               CM.PROV_EXPLICIT_USER),
                  CM.SlotValue(CM.SLOT_ROW_PREDICATES, UNRELATED, "Alpha",
                               CM.PROV_EXPLICIT_USER))
        for binding in (_model_binds(), _model_binds(value="Scotland"),
                        _model_binds(field=UNRELATED, value="Beta")):
            with self.subTest(binding=binding[0].field):
                result = CM.merge(reader, binding, semantics=_semantics())
                held = self._row_slots(result)
                self.assertEqual(held[READER].value, "London")
                self.assertEqual(held[UNRELATED].value, "Alpha")

    def test_the_alias_rule_never_fires_across_different_domains(self):
        """Two fields that do NOT share a value_domain are never one concept,
        so a fill on one can never be suppressed by a claim on the other."""
        from mi_agent.categorical_spans import alias_fields

        sem = _semantics()
        self.assertNotIn(UNRELATED, alias_fields(READER, sem))
        self.assertNotIn(READER, alias_fields(UNRELATED, sem))
