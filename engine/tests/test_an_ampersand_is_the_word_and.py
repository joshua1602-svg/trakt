"""One region written two ways must clean to one key.

THE DEFECT, from the acquired book on 2026-09-03. It carries both
"Yorkshire & Humberside" and "YORKSHIRE AND HUMBERSIDE". The deterministic
cleaner exists so that "every casing and punctuation variant of one region
converges on one key before any mapping is attempted" -- and it did not:

    'Yorkshire & Humberside'    -> 'yorkshire humberside'
    'YORKSHIRE AND HUMBERSIDE'  -> 'yorkshire and humberside'

`_PUNCT_RE` is `[^\\w\\s]+`, so "&" was replaced by a SPACE. Two keys, never
merged, and the region appeared as two rows in every breakdown of that book.
Case was never the problem -- "SOUTH WEST" and "South West" already converge,
and so do "LONDON" and "London" -- the connector was.

An ampersand in a region name is the word "and". Nothing else about the
cleaning changes: other punctuation still becomes a separator, separators still
collapse, and every approved synonym already in `region_taxonomy.yaml` is keyed
on a form carrying no ampersand, so none of them moves. The last test in
`TestNothingElseAboutCleaningMoves` is the falsifiable version of that claim.
"""
from __future__ import annotations

import unittest

from engine import region_taxonomy as RT


class TestTheTwoWritingsConverge(unittest.TestCase):

    def test_the_lenders_own_two_values_clean_to_one_key(self):
        self.assertEqual(RT.clean("Yorkshire & Humberside"),
                         RT.clean("YORKSHIRE AND HUMBERSIDE"))

    def test_the_key_is_the_spelled_out_form(self):
        self.assertEqual(RT.clean("Yorkshire & Humberside"),
                         "yorkshire and humberside")

    def test_spacing_around_the_ampersand_does_not_matter(self):
        for written in ("Yorkshire&Humberside", "Yorkshire & Humberside",
                        "Yorkshire  &  Humberside", "yorkshire and humberside"):
            self.assertEqual(RT.clean(written), "yorkshire and humberside",
                             written)


class TestNothingElseAboutCleaningMoves(unittest.TestCase):
    """The cleaner produces the key for the canonical values AND for every
    approved synonym. A change here that moved any existing key would silently
    unmap a region that resolves today."""

    def test_case_still_converges_as_it_already_did(self):
        self.assertEqual(RT.clean("SOUTH WEST"), RT.clean("South West"))
        self.assertEqual(RT.clean("LONDON"), "london")

    def test_other_punctuation_still_becomes_a_separator(self):
        self.assertEqual(RT.clean("south-west"), "south west")
        self.assertEqual(RT.clean("Yorkshire, Humber"), "yorkshire humber")

    def test_the_governed_canonical_values_keep_their_keys(self):
        self.assertEqual(RT.clean("Yorkshire and The Humber"),
                         "yorkshire and the humber")

    def test_an_absent_value_is_still_absent(self):
        for empty in (None, "", "   ", "nan", "NULL"):
            self.assertEqual(RT.clean(empty), "")

    def test_every_shipped_key_is_its_own_fixed_point(self):
        """'No existing key moves', stated so it can fail: re-clean every
        approved synonym key and every canonical key and require each to come
        back unchanged."""
        taxonomy = RT.resolve_taxonomy(None)
        self.assertIsNotNone(taxonomy)
        for key in list(taxonomy.synonyms) + list(taxonomy.values_by_key):
            self.assertEqual(RT.clean(key), key, key)


class TestItResolvesOnceTheSynonymIsApproved(unittest.TestCase):
    """Cleaning alone is not enough, and this is the half that is governance
    rather than code. "Humberside" is a different word from "Humber", so the
    converged key still has to be an APPROVED synonym before it resolves."""

    def test_both_writings_now_reach_the_governed_canonical(self):
        taxonomy = RT.resolve_taxonomy(None)
        for written in ("Yorkshire & Humberside", "YORKSHIRE AND HUMBERSIDE"):
            value, method = taxonomy.resolve_detail(written)
            self.assertEqual(value, "Yorkshire and The Humber", written)
            self.assertEqual(method, RT.METHOD_SYNONYM, written)

    def test_an_unknown_region_is_still_unresolved_never_guessed(self):
        taxonomy = RT.resolve_taxonomy(None)
        self.assertEqual(taxonomy.resolve_detail("Atlantis"),
                         (None, RT.METHOD_UNRESOLVED))


if __name__ == "__main__":
    unittest.main()
