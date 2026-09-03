"""The bridge's region family must include the harmonised columns.

THE REGRESSION, caught by the 2026-09-03 replay against the deployed build.
"Show movement by region." answered before and refused after, with "I can't
build a funded balance bridge yet: the requested attribution dimension is not
available in the funded data." Its spec named `canonical_region_reporting`.

That naming is the registry change WORKING: `_preferred_region` is data-aware,
so it only picks the harmonised column when that column is present, and it
picked it. What broke is that `chat_routing` keeps a SECOND list of region
columns for the bridge — `_REGION_FAMILY` — and the harmonised pair was not in
it. A concept inside the family resolves to every candidate column so the
bridge can use whichever the funded tape carries; a concept outside it resolves
to one column, and that one was absent from the bridge's own frames.

So the defect is not that MI preferred the harmonised column. It is that one
route had its own idea of what "region" spells as, and adding a governed region
field left it behind. The family is the seam its docstring already describes —
"the bridge picks whichever geography column the funded tape actually carries"
— and it simply has to know about all of them.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from mi_agent_api import chat_routing as CR

_REGISTRY = (Path(__file__).resolve().parents[2] / "mi_agent"
             / "mi_semantics_field_registry.yaml")

REPORTING = "canonical_region_reporting"
DETAIL = "canonical_region_detail"
RAW = "collateral_geography"


def _semantics() -> dict:
    with open(_REGISTRY, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestTheHarmonisedRegionIsInTheFamily(unittest.TestCase):

    def setUp(self):
        self.semantics = _semantics()

    def test_the_harmonised_columns_are_family_members(self):
        for key in (REPORTING, DETAIL):
            self.assertIn(key, CR._REGION_FAMILY, key)

    def test_the_raw_columns_are_still_family_members(self):
        """Adding to the family must not remove anyone: a tape that never
        harmonised still bridges by its own geography column."""
        for key in (RAW, "geographic_region_collateral",
                    "geographic_region_obligor"):
            self.assertIn(key, CR._REGION_FAMILY, key)

    def test_the_harmonised_region_resolves_to_every_candidate_column(self):
        """THE REGRESSION ITSELF. Outside the family this returned one column
        — the harmonised one — and the bridge refused when its own frames did
        not carry it. Inside the family it returns them all, and the bridge
        uses whichever is present."""
        key, cols, label = CR._bridge_dimension(REPORTING, self.semantics)
        self.assertEqual(key, REPORTING)
        self.assertIsInstance(cols, list)
        self.assertIn(RAW, cols)
        self.assertIn(REPORTING, cols)

    def test_the_harmonised_column_is_offered_before_the_raw_one(self):
        _, cols, _ = CR._bridge_dimension(REPORTING, self.semantics)
        self.assertLess(cols.index(REPORTING), cols.index(RAW))

    def test_a_raw_region_concept_still_resolves_to_the_family(self):
        key, cols, _ = CR._bridge_dimension(RAW, self.semantics)
        self.assertEqual(key, RAW)
        self.assertIn(RAW, cols)

    def test_a_non_region_dimension_is_untouched(self):
        key, cols, label = CR._bridge_dimension("broker_channel", self.semantics)
        self.assertEqual(key, "broker_channel")
        self.assertEqual(cols, "broker_channel")
        self.assertEqual(label, "Broker")

    def test_the_label_is_the_business_name_not_the_column(self):
        _, _, label = CR._bridge_dimension(REPORTING, self.semantics)
        self.assertEqual(label, "Region")


class TestTheDefaultDimensionsStillWork(unittest.TestCase):
    """`_BRIDGE_DEFAULT_DIMS` is used when the question names no dimension. It
    is a different list and this change does not touch it."""

    def test_an_unnamed_dimension_still_falls_back_to_a_default(self):
        key, cols, _ = CR._bridge_dimension(None, _semantics())
        self.assertIn(key, CR._BRIDGE_DEFAULT_DIMS)
        self.assertTrue(cols)

    def test_an_unknown_concept_falls_back_rather_than_failing(self):
        key, cols, _ = CR._bridge_dimension("not_a_field", _semantics())
        self.assertIn(key, CR._BRIDGE_DEFAULT_DIMS)
        self.assertTrue(cols)


if __name__ == "__main__":
    unittest.main()
