"""The harmonised region column is what a region question should read.

WHAT THE 2026-09-03 AUDIT ESTABLISHED. `engine.region_taxonomy` runs on the
platform-canonical path and stamps `canonical_region_detail` /
`canonical_region_reporting` onto the frame — 82.6% of the acquired book's
region values and 90% of the direct book's resolve to a governed canonical.
None of that reached a reader. The two columns were not in the semantics
registry, so MI had no governed field for them, could not group or filter by
them, and every region question bound to the RAW source column instead. In the
acquired book that column carries both "LONDON" and "London", so one region
came back as two rows in every breakdown, and each row carried half the book.

`region_taxonomy`'s own docstring says "the runtime MI path — queries, charts,
the geography view, exports — reads the persisted canonical columns". It did
not. The columns were written and nothing was registered to read them.

TWO THINGS MAKE THAT TRUE, and both are pinned here.

  * The columns are REGISTERED, so a governed field exists for them.
  * `_preferred_region` puts the harmonised field first. That function is
    already data-aware — it walks a preference order and takes the first field
    whose column is actually PRESENT — so a dataset that never harmonised is
    unaffected and keeps `collateral_geography`, which is what every existing
    fixture does.

WHAT MUST NOT HAPPEN. The generic word "region" must not become ambiguous.
`_registry_dimension_terms` DROPS any synonym mapping to more than one
dimension, so giving the canonical fields the generic vocabulary would delete
the term and refuse every region question. The generic terms route through
`_preferred_region` by design; the canonical entries carry only names specific
to themselves.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from mi_agent import llm_query_parser as P

_REGISTRY = Path(__file__).resolve().parents[1] / "mi_semantics_field_registry.yaml"

DETAIL = "canonical_region_detail"
REPORTING = "canonical_region_reporting"
RAW = "collateral_geography"


def _semantics() -> dict:
    with open(_REGISTRY, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestTheColumnsAreRegistered(unittest.TestCase):

    def setUp(self):
        self.fields = _semantics()["fields"]

    def test_both_harmonised_columns_are_governed_fields(self):
        for key in (DETAIL, REPORTING):
            self.assertIn(key, self.fields,
                          "%s is stamped on the frame but nothing can read it" % key)

    def test_they_are_dimensions_a_question_can_group_by(self):
        for key in (DETAIL, REPORTING):
            self.assertEqual(self.fields[key]["role"], "dimension", key)
            self.assertTrue(self.fields[key].get("chartable"), key)

    def test_they_carry_the_uk_region_value_domain(self):
        """The domain the value resolver uses to turn "Scotland" in a question
        into the value the column holds."""
        for key in (DETAIL, REPORTING):
            self.assertEqual(self.fields[key].get("value_domain"), "uk_region", key)

    def test_they_declare_what_they_are_derived_from(self):
        for key in (DETAIL, REPORTING):
            self.assertTrue(self.fields[key].get("derived"), key)


class TestTheHarmonisedColumnWinsWhenItIsThere(unittest.TestCase):

    def setUp(self):
        self.semantics = _semantics()

    def test_a_harmonised_dataset_reads_the_harmonised_column(self):
        chosen = P._preferred_region(
            self.semantics, available_columns={RAW, DETAIL, REPORTING})
        self.assertEqual(chosen, REPORTING)

    def test_the_reporting_column_is_preferred_over_the_detail_one(self):
        """`canonical_region_reporting` is the client-level value used when
        books are combined, and equals the detail value unless the client's
        approved taxonomy declares a consolidation. The detail column is
        retained beside it and is reachable by name."""
        self.assertLess(P._REGION_PREFERENCE.index(REPORTING),
                        P._REGION_PREFERENCE.index(DETAIL))

    def test_a_dataset_that_never_harmonised_is_unaffected(self):
        """Every existing fixture. The preference is data-aware, so adding a
        field ahead of `collateral_geography` changes nothing where the
        harmonised column is absent."""
        chosen = P._preferred_region(self.semantics, available_columns={RAW})
        self.assertEqual(chosen, RAW)

    def test_the_raw_column_still_wins_over_the_nuts_code_fields(self):
        chosen = P._preferred_region(
            self.semantics,
            available_columns={RAW, "geographic_region_obligor"})
        self.assertEqual(chosen, RAW)


class TestTheWordRegionDoesNotBecomeAmbiguous(unittest.TestCase):
    """`_registry_dimension_terms` drops a synonym that maps to more than one
    dimension. Giving the canonical fields the generic region vocabulary would
    therefore DELETE the term and refuse every region question — the opposite
    of the intent."""

    def setUp(self):
        self.semantics = _semantics()
        self.fields = self.semantics["fields"]

    def test_the_canonical_fields_claim_no_generic_region_term(self):
        for key in (DETAIL, REPORTING):
            claimed = {str(s).strip().lower()
                       for s in (self.fields[key].get("synonyms") or [])}
            self.assertFalse(claimed & P._REGION_GENERIC_TERMS,
                             "%s claims a generic term routed through "
                             "_preferred_region: %s"
                             % (key, sorted(claimed & P._REGION_GENERIC_TERMS)))

    def test_the_generic_terms_still_resolve_to_a_region_field(self):
        terms = P._registry_dimension_terms(self.semantics)
        terms.update(P.EXPLICIT_DIMENSION_TERMS)
        for generic in ("region", "regional", "geographic region"):
            self.assertIn(terms.get(generic), set(P._REGION_PREFERENCE),
                          "%r no longer names a region dimension" % generic)

    def test_no_other_dimension_lost_its_vocabulary_to_these_two(self):
        """The blast radius of adding registry vocabulary, stated so it can
        fail: every term the registry resolved before must still resolve, and
        to the same field."""
        terms = P._registry_dimension_terms(self.semantics)
        for key in (DETAIL, REPORTING):
            for phrase in (self.fields[key].get("synonyms") or []):
                owner = terms.get(str(phrase).strip().lower())
                self.assertIn(owner, (None, key),
                              "%r was taken from %s" % (phrase, owner))


if __name__ == "__main__":
    unittest.main()
